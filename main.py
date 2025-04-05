import gradio as gr
import torch
import random
import os
import datetime
from PIL import ImageFont

from utils.gradio_utils import (
    character_to_dict,
    process_original_prompt,
    get_ref_character,
    is_torch2_available,
    AttnProcessor,
)
from utils.utils import get_comic
from utils.style_template import styles
from utils.load_models_utils import get_models_dict, load_models

# ====== Constants and Globals ======
device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEED = 2147483647
DEFAULT_STYLE_NAME = "Japanese Anime"
STYLE_NAMES = list(styles.keys())
models_dict = get_models_dict()
pipe = None

# ====== Load Comic Model ======
model_name = "ComicModel"
model_info = models_dict[model_name]
pipe = load_models(model_info, device=device)
pipe.to(device)
pipe.enable_vae_slicing()
pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
if device != "mps":
    pipe.enable_model_cpu_offload()

# ====== Core Generation ======
def apply_style_positive(style_name: str, positive: str):
    p, _ = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive)

def apply_style(style_name: str, positives: list, negative: str = ""):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return [p.replace("{prompt}", pos) for pos in positives], n + " " + negative

def setup_seed(seed):
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def process_generation(
    seed,
    style_name,
    general_prompt,
    prompt_array,
    font_choice,
    steps,
    width,
    height,
    guidance_scale,
    comic_type
):
    setup_seed(seed)
    character_dict, character_list = character_to_dict(general_prompt)
    prompts = prompt_array.strip().splitlines()
    
    prompts_clean = [line.split("#")[0].replace("[NC]", "").strip() for line in prompts]
    captions = [line.split("#")[-1].strip() if "#" in line else "" for line in prompts]
    character_index_dict, _, processed_prompts, _, _ = process_original_prompt(character_dict, prompts_clean, 0)

    final_images = []
    for prompt in processed_prompts:
        styled_prompt = apply_style_positive(style_name, prompt)
        result = pipe(
            styled_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width
        ).images[0]
        final_images.append(result)

    # Apply comic layout
    font_path = os.path.join("fonts", font_choice)
    font = ImageFont.truetype(font_path, 40)
    comic_images = get_comic(final_images, comic_type, captions, font)

    # Save output
    output_dir = f"output/out_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    for idx, img in enumerate(comic_images):
        img.save(f"{output_dir}/img{idx + 1}.png")

    return comic_images


# ====== Gradio UI ======
with gr.Blocks(title="NarrativeDiffusion") as demo:
    gr.Markdown("## 🎨 NarrativeDiffusion: Generate Comic-Style Story Panels")

    with gr.Row():
        with gr.Column():
            general_prompt = gr.Textbox(label="Character Descriptions", lines=2, placeholder="[Tom] a boy with a red cape")
            prompt_array = gr.Textbox(label="Story Prompts", lines=6, placeholder="[Tom] runs into the woods. #He looks around nervously.")
            style_dropdown = gr.Dropdown(choices=STYLE_NAMES, value=DEFAULT_STYLE_NAME, label="Art Style")
            font_choice = gr.Dropdown(choices=[f for f in os.listdir("fonts") if f.endswith(".ttf")], value="Inkfree.ttf", label="Font")
            steps = gr.Slider(20, 50, value=30, step=1, label="Sampling Steps")
            guidance = gr.Slider(1.0, 10.0, value=5.0, step=0.5, label="Guidance Scale")
            width = gr.Slider(512, 1024, step=64, value=768, label="Image Width")
            height = gr.Slider(512, 1024, step=64, value=768, label="Image Height")
            comic_type = gr.Radio(["Classic Comic Style", "Four Pannel", "No typesetting (default)"], value="Classic Comic Style", label="Layout")
            seed = gr.Slider(0, MAX_SEED, value=0, step=1, label="Seed")
            run_button = gr.Button("Generate Story 🎬")

        with gr.Column():
            gallery = gr.Gallery(label="Generated Comic", columns=2, height="auto")

    run_button.click(
        fn=process_generation,
        inputs=[
            seed,
            style_dropdown,
            general_prompt,
            prompt_array,
            font_choice,
            steps,
            width,
            height,
            guidance,
            comic_type
        ],
        outputs=gallery
    )

demo.launch(share=True)
