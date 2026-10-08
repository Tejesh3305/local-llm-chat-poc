"""Real image generation with SD-Turbo (Stability AI), a small text-to-image model that fits a 6 GB GPU.

Optional: it needs PyTorch, diffusers and the model files (see requirements-images.txt and
scripts/get_image_model.py). Without them, AI Studio falls back to its demo drawings.

The GPU is shared with the chat model, so a picture is drawn while the chat model is unloaded
(the caller holds the registry's slot), and the image model is moved back to system memory afterwards.
"""

import importlib.util
import io
import logging
import threading
import time
from pathlib import Path

logger = logging.getLogger("app.imagegen")

MODEL_NAME = "SD-Turbo"
SIZE = 512  # what SD-Turbo was trained for; other sizes give worse pictures

# Words added to the prompt for each style the page offers
STYLE_HINTS = {
    "photo": "high quality photograph, natural light, sharp focus, detailed",
    "landscape": "landscape photograph, natural light, wide angle, highly detailed",
    "terrain map": "topographic map seen from above, contour lines, cartographic style, muted colours",
    "poster": "flat vector poster, bold simple shapes, strong colours, graphic design",
}


class ImageGenerator:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self._pipe = None
        self._device: str | None = None  # "GPU" or "CPU (slow)", known once PyTorch is imported
        self._lock = threading.Lock()  # one picture at a time; also guards loading

    def status(self) -> dict:
        """Whether real generation can run here, and if not, why not (in words for the page).

        Fast: it only looks for the packages and files. Importing PyTorch takes seconds, so that is left to warm_up().
        """
        if not (self.model_dir / "model_index.json").exists():
            return {"available": False, "model": MODEL_NAME, "reason": "The image model is not downloaded. Run: python scripts/get_image_model.py"}
        if not (importlib.util.find_spec("torch") and importlib.util.find_spec("diffusers")):
            return {"available": False, "model": MODEL_NAME, "reason": "PyTorch and diffusers are not installed. Run: pip install -r requirements-images.txt"}
        return {"available": True, "model": MODEL_NAME, "device": self._device, "reason": ""}

    def warm_up(self) -> None:
        """Import PyTorch in the background, so the first picture and the device name do not wait for it."""
        if self.status()["available"]:
            threading.Thread(target=self._probe, name="image-model-warm-up", daemon=True).start()

    def _probe(self) -> None:
        import torch

        self._device = "GPU" if torch.cuda.is_available() else "CPU (slow)"

    def _load(self):
        if self._pipe is None:
            import torch
            from diffusers import AutoPipelineForText2Image
            from diffusers.utils import logging as diffusers_logging

            # Parking the half-precision model in system memory between pictures makes diffusers warn that it
            # cannot run there. It never runs there, so only errors are shown.
            diffusers_logging.set_verbosity_error()
            started = time.perf_counter()
            # Only the half-precision ("fp16") weights are downloaded; on a CPU they are widened to full precision
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self._pipe = AutoPipelineForText2Image.from_pretrained(str(self.model_dir), dtype=dtype, variant="fp16", local_files_only=True)
            self._pipe.set_progress_bar_config(disable=True)
            logger.info("loaded %s in %.1f s", MODEL_NAME, time.perf_counter() - started)
        return self._pipe

    def generate(self, prompt: str, style: str, seed: int, steps: int = 2) -> tuple[bytes, dict]:
        """A PNG of the prompt. Blocking: call it in a worker thread, while holding the model slot."""
        import torch

        with self._lock:
            pipe = self._load()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = "GPU" if device == "cuda" else "CPU (slow)"
            full_prompt = f"{prompt.strip()}, {STYLE_HINTS.get(style, '')}".strip(", ")
            started = time.perf_counter()
            try:
                pipe.to(device)
                generator = torch.Generator(device=device).manual_seed(seed)
                # SD-Turbo is distilled to work without guidance in 1 to 4 steps
                image = pipe(prompt=full_prompt, num_inference_steps=steps, guidance_scale=0.0, width=SIZE, height=SIZE, generator=generator).images[0]
            finally:
                # Give the GPU back to the chat model
                pipe.to("cpu")
                if device == "cuda":
                    torch.cuda.empty_cache()
            seconds = round(time.perf_counter() - started, 1)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue(), {"prompt_used": full_prompt, "steps": steps, "seed": seed, "seconds": seconds, "width": SIZE, "height": SIZE}
