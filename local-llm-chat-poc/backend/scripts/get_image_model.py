"""Download SD-Turbo, the image model used by AI Studio, into backend/models/sd-turbo (about 2.5 GB).

    python scripts/get_image_model.py

Run it once while online. Afterwards the app loads the model from that folder and needs no internet.
Only the half-precision (fp16) weights are fetched, which is all a GPU needs.
"""

from pathlib import Path

from huggingface_hub import snapshot_download

TARGET = Path(__file__).resolve().parent.parent / "models" / "sd-turbo"

if __name__ == "__main__":
    print(f"Downloading stabilityai/sd-turbo to {TARGET} ...")
    snapshot_download(
        "stabilityai/sd-turbo",
        local_dir=TARGET,
        allow_patterns=["model_index.json", "*.txt", "*/config.json", "*/*config.json", "*/*.fp16.safetensors", "tokenizer/*", "scheduler/*", "*.md"],
    )
    print("Done. Restart the backend; AI Studio's Image generation now uses SD-Turbo.")
