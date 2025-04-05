from diffusers import StableDiffusionXLPipeline
import torch

def get_models_dict():
    return {
        "ComicModel": {
            "path": "SG161222/RealVisXL_V4.0",
            "single_files": False
        }
    }

def load_models(model_info, device="cuda"):
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_info["path"],
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    return pipe
