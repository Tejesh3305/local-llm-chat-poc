"""Download OWLv2 (Google), the object detection model used by the chat and AI Studio, into
backend/models/owlv2 (about 600 MB).

    python scripts/get_detection_model.py

Run it once while online. Afterwards the app loads the model from that folder and needs no internet.
"""

from pathlib import Path

from huggingface_hub import snapshot_download

TARGET = Path(__file__).resolve().parent.parent / "models" / "owlv2"

if __name__ == "__main__":
    print(f"Downloading google/owlv2-base-patch16-ensemble to {TARGET} ...")
    snapshot_download("google/owlv2-base-patch16-ensemble", local_dir=TARGET, allow_patterns=["*.json", "*.txt", "*.safetensors", "*.md", "LICENSE*"])
    print("Done. Restart the backend; object detection is then available in the chat and AI Studio.")
