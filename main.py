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

#Constants and Globals
device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEED = 2147483647
DEFAULT_STYLE_NAME = "Japanese Anime"
STYLE_NAMES = list(styles.keys())
models_dict = get_models_dict()
pipe = None

# State
gallery_images = []
caption_texts = []
character_dict = {}
processed_prompts = []
original_prompts = []
character_registry = {}
current_style_name = DEFAULT_STYLE_NAME
current_character_input = ""

# Load Comic Model
model_name = "ComicModel"
model_info = models_dict[model_name]
pipe = load_models(model_info, device=device)
pipe.to(device)
pipe.enable_vae_slicing()
pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
if device != "mps":
    pipe.enable_model_cpu_offload()

# Utils
def apply_style_positive(style_name: str, positive: str):
    p, _ = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive)

def setup_seed(seed):
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def update_character_registry(general_prompt):
    global character_registry
    char_dict, _ = character_to_dict(general_prompt)
    for tag, desc in char_dict.items():
        if tag not in character_registry:
            character_registry[tag] = {"base": desc, "traits": []}
    return char_dict

def get_full_character_desc(tag):
    entry = character_registry.get(tag, {})
    base = entry.get("base", "")
    traits = ", ".join(entry.get("traits", []))
    return f"{base}, {traits}" if traits else base

# Main Generation
def process_generation(seed, style_name, general_prompt, prompt_array, font_choice,
                       steps, width, height, guidance_scale, comic_type):
    global gallery_images, caption_texts, original_prompts, character_dict, processed_prompts
    global current_style_name, current_character_input
    try:
        current_style_name = style_name
        current_character_input = general_prompt
        setup_seed(seed)
        character_dict = update_character_registry(general_prompt)

        prompts_raw = prompt_array.strip().splitlines()
        prompts_clean = [line.split("#")[0].replace("[NC]", "").strip() for line in prompts_raw]
        caption_texts[:] = [line.split("#")[-1].strip() if "#" in line else "" for line in prompts_raw]

        _, _, processed_prompts, _, _ = process_original_prompt(character_dict, prompts_clean, 0)
        original_prompts[:] = processed_prompts.copy()

        gallery_images.clear()
        for prompt in processed_prompts:
            styled_prompt = apply_style_positive(style_name, prompt)
            result = pipe(styled_prompt, num_inference_steps=steps,
                          guidance_scale=guidance_scale,
                          height=height, width=width).images[0]
            gallery_images.append(result)

        font_path = os.path.join("fonts", font_choice)
        font = ImageFont.truetype(font_path, 40)
        comic_images = get_comic(gallery_images, comic_type, caption_texts, font)

        output_dir = f"output/out_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        os.makedirs(output_dir, exist_ok=True)
        for idx, img in enumerate(comic_images):
            img.save(f"{output_dir}/img{idx + 1}.png")

        panel_choices = [str(i) for i in range(len(gallery_images))]
        return comic_images, gr.update(choices=panel_choices, value=panel_choices[-1]), "", ""

    except Exception as e:
        return gallery_images, gr.update(), refine_prompt, f"Error in generation: {str(e)}"

# Add Scene
def add_new_scene(new_scene_prompt, font_choice, steps, width, height, guidance, comic_type):
    global gallery_images, processed_prompts, caption_texts, current_character_input, current_style_name
    try:
        if not new_scene_prompt.strip():
            return gallery_images, gr.update(), "", "Enter a valid scene prompt"

        character_dict = update_character_registry(current_character_input)
        prompt = new_scene_prompt.split("#")[0].replace("[NC]", "").strip()
        caption = new_scene_prompt.split("#")[-1].strip() if "#" in new_scene_prompt else ""
        _, _, processed, _, _ = process_original_prompt(character_dict, [prompt], 0)
        styled_prompt = apply_style_positive(current_style_name, processed[0])

        setup_seed(random.randint(0, MAX_SEED))
        new_image = pipe(styled_prompt, num_inference_steps=steps,
                         guidance_scale=guidance, height=height, width=width).images[0]

        gallery_images.append(new_image)
        processed_prompts.append(prompt)
        caption_texts.append(caption)

        font_path = os.path.join("fonts", font_choice)
        font = ImageFont.truetype(font_path, 40)
        comic_images = get_comic(gallery_images, comic_type, caption_texts, font)

        panel_choices = [str(i) for i in range(len(gallery_images))]
        return comic_images, gr.update(choices=panel_choices, value=panel_choices[-1]), "", ""

    except Exception as e:
        return gallery_images, gr.update(), "", f"Error: {str(e)}"

# Feedback Refinement
def refine_panel(index, refinement_text, style_name, font_choice, steps, width, height, guidance, comic_type):
    global gallery_images, processed_prompts, caption_texts, current_character_input
    try:
        index = int(index)
        base_prompt = processed_prompts[index]

        character_dict = update_character_registry(current_character_input)

        full_prompt = base_prompt
        if refinement_text.strip():
            full_prompt += f", {refinement_text.strip()}"

        _, _, processed, _, _ = process_original_prompt(character_dict, [full_prompt], 0)
        styled_prompt = apply_style_positive(style_name, processed[0])

        setup_seed(random.randint(0, MAX_SEED))
        new_image = pipe(
            styled_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=height,
            width=width
        ).images[0]

        gallery_images[index] = new_image

        font_path = os.path.join("fonts", font_choice)
        font = ImageFont.truetype(font_path, 40)
        comic_images = get_comic(gallery_images, comic_type, caption_texts, font)

        return comic_images, "", ""

    except Exception as e:
        return gallery_images, "", f"Error: {str(e)}"

# Gradio UI
with gr.Blocks(title="NarrativeDiffusion") as demo:
    gr.Markdown("## 🎨 NarrativeDiffusion: Comic Story Generator with Feedback Refinement")

    with gr.Row():
        with gr.Column():
            general_prompt = gr.Textbox(label="Define Characters", lines=3,
                placeholder="[Tom] a boy with a red cape\n[Emma] a girl with silver hair")
            prompt_array = gr.Textbox(label="Story Prompts", lines=6,
                placeholder="[Tom] walks into the forest. #He is scared.")
            style_dropdown = gr.Dropdown(choices=STYLE_NAMES, 
                value=DEFAULT_STYLE_NAME, label="Art Style")
            font_choice = gr.Dropdown(choices=[f for f in os.listdir("fonts") if f.endswith(".ttf")], 
                value="Inkfree.ttf", label="Font")
            steps = gr.Slider(20, 50, value=30, label="Sampling Steps")
            guidance = gr.Slider(1.0, 10.0, value=5.0, label="Guidance Scale")
            width = gr.Slider(512, 1024, step=64, value=768, label="Image Width")
            height = gr.Slider(512, 1024, step=64, value=768, label="Image Height")
            comic_type = gr.Radio(["Classic Comic Style", "Four Pannel", "No typesetting (default)"], 
                value="Classic Comic Style", label="Layout")
            seed = gr.Slider(0, MAX_SEED, value=0, label="Seed")
            run_button = gr.Button("🖼️ Generate Story")

        with gr.Column():
            gallery = gr.Gallery(label="Generated Comic Panels", columns=2, height="auto")
            panel_selector = gr.Dropdown(choices=[], label="Select Panel to Refine")
            refine_prompt = gr.Textbox(label="Add Trait (e.g. 'has a red scarf')", placeholder="Trait or appearance change")
            refine_btn = gr.Button("♻️ Refine Panel")
            gr.Markdown("---")
            new_scene_input = gr.Textbox(label="➕ Add a New Scene", placeholder="[Tom] enters the cave. #It's dark.")
            add_scene_btn = gr.Button("Add Scene ➕")
            error_box = gr.Textbox(label="Error Messages", interactive=False)

    run_button.click(
        fn=process_generation,
        inputs=[seed, style_dropdown, general_prompt, prompt_array, font_choice, steps, width, height, guidance, comic_type],
        outputs=[gallery, panel_selector, refine_prompt, error_box]
    )

    refine_btn.click(
       fn=refine_panel,
       inputs=[panel_selector, refine_prompt, style_dropdown, font_choice, steps, width, height, guidance, comic_type],
       outputs=[gallery, panel_selector, error_box]
    )

    add_scene_btn.click(
        fn=add_new_scene,
        inputs=[new_scene_input, font_choice, steps, width, height, guidance, comic_type],
        outputs=[gallery, panel_selector, refine_prompt, error_box]
    )

demo.launch(share=True)
