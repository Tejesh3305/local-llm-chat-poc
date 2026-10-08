"""Object detection with OWLv2 (Google, Apache-2.0): finds the objects named in plain words and boxes them.

OWLv2 is "open vocabulary": it is given the names of the objects to look for, so it can find military
equipment that everyday detectors (trained on 80 household classes) do not know. Optional, like the image
model: it needs PyTorch, transformers and the model files (scripts/get_detection_model.py).

Accuracy is demo grade. Small, blurred or partly hidden objects are missed, and similar-looking vehicles
can be mixed up; reliable military detection needs a model trained on labelled photos of the equipment.
"""

import importlib.util
import io
import logging
import re
import threading
import time
from pathlib import Path

logger = logging.getLogger("app.vision")

MODEL_NAME = "OWLv2"

# Looked for when the message names nothing in particular ("detect objects in this image").
# Military first, then everyday objects. Each entry: the label shown, and the words given to the model.
DEFAULT_TARGETS: list[tuple[str, list[str]]] = [
    ("tank", ["a tank", "an armoured tank"]),
    ("armoured vehicle", ["an armoured personnel carrier", "an armoured military vehicle"]),
    ("military truck", ["a military truck", "an army truck"]),
    ("artillery", ["an artillery gun", "a howitzer"]),
    ("missile launcher", ["a missile launcher", "a rocket launcher"]),
    ("helicopter", ["a helicopter"]),
    ("fighter jet", ["a fighter jet", "a military aircraft"]),
    ("airplane", ["an airplane"]),
    ("drone", ["a drone", "a quadcopter"]),
    ("soldier", ["a soldier", "a person in military uniform"]),
    ("person", ["a person"]),
    ("rifle", ["a rifle", "a gun"]),
    ("warship", ["a warship"]),
    ("boat", ["a boat"]),
    ("car", ["a car"]),
    ("jeep", ["a jeep"]),
    ("truck", ["a truck"]),
    ("bus", ["a bus"]),
    ("motorcycle", ["a motorcycle"]),
    ("bicycle", ["a bicycle"]),
    ("tent", ["a tent"]),
    ("bunker", ["a bunker", "a sandbag position"]),
    ("building", ["a building"]),
    ("bridge", ["a bridge"]),
    ("road", []),  # never boxed: too large to be useful
]
# Words in a message -> the target they mean (plurals and common variants included)
ALIASES = {
    "tank": "tank", "tanks": "tank", "mbt": "tank",
    "apc": "armoured vehicle", "apcs": "armoured vehicle", "ifv": "armoured vehicle", "armoured": "armoured vehicle", "armored": "armoured vehicle",
    "military truck": "military truck", "military trucks": "military truck", "army truck": "military truck", "army trucks": "military truck",
    "armoured vehicle": "armoured vehicle", "armoured vehicles": "armoured vehicle", "armored vehicle": "armoured vehicle", "armored vehicles": "armoured vehicle",
    "military vehicle": "armoured vehicle", "military vehicles": "armoured vehicle", "fighter jet": "fighter jet", "fighter jets": "fighter jet",
    "missile launcher": "missile launcher", "missile launchers": "missile launcher", "rocket launcher": "missile launcher", "rocket launchers": "missile launcher",
    "artillery": "artillery", "howitzer": "artillery", "howitzers": "artillery", "cannon": "artillery", "guns": "artillery",
    "launcher": "missile launcher", "launchers": "missile launcher", "missile": "missile launcher", "missiles": "missile launcher", "rocket": "missile launcher",
    "helicopter": "helicopter", "helicopters": "helicopter", "chopper": "helicopter", "choppers": "helicopter",
    "jet": "fighter jet", "jets": "fighter jet", "fighter": "fighter jet", "fighters": "fighter jet",
    "airplane": "airplane", "airplanes": "airplane", "aircraft": "airplane", "plane": "airplane", "planes": "airplane",
    "drone": "drone", "drones": "drone", "uav": "drone", "uavs": "drone",
    "soldier": "soldier", "soldiers": "soldier", "troops": "soldier", "troop": "soldier", "personnel": "soldier",
    "person": "person", "people": "person", "persons": "person", "human": "person", "humans": "person", "man": "person", "men": "person",
    "rifle": "rifle", "rifles": "rifle", "gun": "rifle", "weapon": "rifle", "weapons": "rifle",
    "warship": "warship", "warships": "warship", "ship": "warship", "ships": "warship",
    "boat": "boat", "boats": "boat",
    "car": "car", "cars": "car", "vehicle": "car", "vehicles": "car",
    "jeep": "jeep", "jeeps": "jeep",
    "truck": "truck", "trucks": "truck", "lorry": "truck", "lorries": "truck",
    "bus": "bus", "buses": "bus",
    "motorcycle": "motorcycle", "motorcycles": "motorcycle", "bike": "motorcycle", "bikes": "motorcycle",
    "bicycle": "bicycle", "bicycles": "bicycle",
    "tent": "tent", "tents": "tent", "camp": "tent",
    "bunker": "bunker", "bunkers": "bunker",
    "building": "building", "buildings": "building", "house": "building", "houses": "building",
    "bridge": "bridge", "bridges": "bridge",
}
# "vehicles" also means the military ones
GROUPS = {"car": ["car", "jeep", "truck", "military truck", "tank", "armoured vehicle", "bus"], "rifle": ["rifle", "artillery"]}
COLOURS = ["#e8b04a", "#7cc76f", "#8db4e8", "#ff8f84", "#c9a0ff", "#6fc2a6", "#f2c15b", "#ffffff"]


def targets_for(message: str) -> tuple[list[tuple[str, list[str]]], bool]:
    """The objects to look for in this message, and whether they were named (False: the default list)."""
    text = message.lower()
    hits: list[tuple[int, str]] = []  # (position in the message, target), so the order of the message is kept
    for phrase in sorted(ALIASES, key=len, reverse=True):  # "military truck" before "truck"
        for match in re.finditer(rf"\b{re.escape(phrase)}\b", text):
            label = ALIASES[phrase]
            group = GROUPS.get(label, [label]) if phrase in ("vehicle", "vehicles", "weapon", "weapons") else [label]
            hits += [(match.start() + n / 100, target) for n, target in enumerate(group)]
        text = re.sub(rf"\b{re.escape(phrase)}\b", lambda found: " " * len(found.group(0)), text)  # same length: positions stay
    named: list[str] = []
    for _, target in sorted(hits):
        if target not in named:
            named.append(target)
    known = dict(DEFAULT_TARGETS)
    if named:
        return [(label, known.get(label) or [f"a {label}"]) for label in named], True
    # Something not in the list ("detect the red flags"): ask the model for exactly that
    free = re.search(r"\b(?:detect|find|locate|spot|count|mark|identify|highlight|box)\s+(?:all\s+|the\s+|any\s+|every\s+)*(.+?)(?:\s+in\s+(?:this|the)\b.*|\s+(?:here|please))?[?.!]*$", message, re.I)
    if free:
        phrase = free.group(1).strip()
        if phrase and not re.fullmatch(r"(objects?|things?|everything|items?|it|them)", phrase, re.I):
            singular = phrase[:-1] if phrase.endswith("s") and not phrase.endswith("ss") else phrase
            return [(singular, [f"a {singular}"])], True
    return [(label, words) for label, words in DEFAULT_TARGETS if words], False


def _iou(a: list[float], b: list[float]) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class Detector:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self._model = None
        self._processor = None
        self._lock = threading.Lock()

    def status(self) -> dict:
        if not (self.model_dir / "config.json").exists():
            return {"available": False, "model": MODEL_NAME, "reason": "The detection model is not downloaded. Run: python scripts/get_detection_model.py"}
        if not (importlib.util.find_spec("torch") and importlib.util.find_spec("transformers")):
            return {"available": False, "model": MODEL_NAME, "reason": "PyTorch and transformers are not installed. Run: pip install -r requirements-images.txt"}
        return {"available": True, "model": MODEL_NAME, "reason": ""}

    def _load(self):
        if self._model is None:
            from transformers import Owlv2ForObjectDetection, Owlv2Processor

            started = time.perf_counter()
            self._processor = Owlv2Processor.from_pretrained(str(self.model_dir), local_files_only=True)
            self._model = Owlv2ForObjectDetection.from_pretrained(str(self.model_dir), local_files_only=True).eval()
            logger.info("loaded %s in %.1f s", MODEL_NAME, time.perf_counter() - started)
        return self._model, self._processor

    def detect(self, image_bytes: bytes, targets: list[tuple[str, list[str]]], threshold: float = 0.2) -> tuple[bytes, list[dict], dict]:
        """Boxes for the targets: the picture with the boxes drawn (PNG), the detections, and timing. Blocking."""
        import torch
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        queries = [words for _, words in targets for words in words]
        label_of = [label for label, words in targets for _ in words]
        with self._lock:
            model, processor = self._load()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            started = time.perf_counter()
            try:
                model.to(device)
                inputs = processor(text=[queries], images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                # OWLv2 works on the picture padded to a square, so the boxes are scaled to that square
                side = max(image.size)
                result = processor.post_process_grounded_object_detection(outputs=outputs, threshold=threshold, target_sizes=[(side, side)])[0]
                found = [
                    {"label": label_of[int(index)], "score": round(float(score), 2), "box": [round(max(0.0, min(float(v), limit)), 1) for v, limit in zip(box, (image.width, image.height) * 2)]}
                    for score, index, box in zip(result["scores"], result["labels"], result["boxes"])
                ]
            finally:
                model.to("cpu")  # give the GPU back to the chat model
                if device == "cuda":
                    torch.cuda.empty_cache()
            seconds = round(time.perf_counter() - started, 1)

        # A weak box next to a strong one of the same kind is usually wrong: keep boxes at least half as sure as the best
        best = {label: max(d["score"] for d in found if d["label"] == label) for label in {d["label"] for d in found}}
        found = [d for d in found if d["score"] >= best[d["label"]] * 0.5]
        # One box per object: drop boxes that mostly overlap a better one (also across labels: "soldier" and "person")
        kept: list[dict] = []
        for item in sorted(found, key=lambda d: -d["score"]):
            if item["box"][2] - item["box"][0] < 4 or item["box"][3] - item["box"][1] < 4:
                continue
            if all(_iou(item["box"], other["box"]) < 0.5 for other in kept):
                kept.append(item)
        kept = kept[:40]
        return self.draw(image, kept), kept, {"seconds": seconds, "device": "GPU" if device == "cuda" else "CPU", "width": image.width, "height": image.height}

    @staticmethod
    def draw(image, detections: list[dict]) -> bytes:
        from PIL import ImageDraw, ImageFont

        canvas = image.copy()
        pen = ImageDraw.Draw(canvas)
        size = max(14, round(max(canvas.size) / 50))
        try:
            font = ImageFont.truetype("arial.ttf", size)
        except OSError:
            font = ImageFont.load_default()
        labels = sorted({d["label"] for d in detections})
        for item in detections:
            colour = COLOURS[labels.index(item["label"]) % len(COLOURS)]
            x0, y0, x1, y1 = item["box"]
            pen.rectangle([x0, y0, x1, y1], outline=colour, width=max(2, size // 6))
            text = f"{item['label']} {round(item['score'] * 100)}%"
            left, top, right, bottom = pen.textbbox((x0, y0), text, font=font)
            lift = bottom - top + 6
            ty = y0 - lift if y0 - lift > 0 else y0
            pen.rectangle([x0, ty, x0 + right - left + 8, ty + lift], fill=colour)
            pen.text((x0 + 4, ty + 2), text, fill="#111111", font=font)
        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")
        return buffer.getvalue()


def summary_line(detections: list[dict]) -> str:
    """'2 tanks, 1 military truck' for the reply."""
    counts: dict[str, int] = {}
    for item in detections:
        counts[item["label"]] = counts.get(item["label"], 0) + 1
    return ", ".join(f"{count} {label}{'s' if count > 1 and not label.endswith('s') else ''}" for label, count in sorted(counts.items(), key=lambda kv: -kv[1]))
