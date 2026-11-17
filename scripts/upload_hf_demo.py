"""Upload the front-only demo pack and Gradio Space to Hugging Face."""

from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

api = HfApi()
root = Path(__file__).resolve().parents[1]
user = api.whoami()["name"]
model_id = f"{user}/garment-routing-front"
space_id = f"{user}/garment-routing"

api.create_repo(model_id, repo_type="model", exist_ok=True, private=False)
api.upload_folder(folder_path=str(root / "demo" / "hf_model"), repo_id=model_id, repo_type="model")
print(f"model https://huggingface.co/{model_id}")

try:
    api.create_repo(space_id, repo_type="space", exist_ok=True, space_sdk="gradio", private=False)
    api.upload_folder(folder_path=str(root / "demo" / "space"), repo_id=space_id, repo_type="space")
    print(f"space https://huggingface.co/spaces/{space_id}")
except HfHubHTTPError as exc:
    print(f"space_upload_failed {exc}")
    print(
        "Gradio Spaces on cpu-basic currently need a Hugging Face PRO subscription. "
        "Model weights are public; run the local demo from demo/space when Gradio is installed."
    )
