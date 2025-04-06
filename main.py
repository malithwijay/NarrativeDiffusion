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

# ===== Constants and Globals =====
device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEED = 2147483647
DEFAULT_STYLE_NAME = "Japanese Anime"
STYLE_NAMES = list(styles.keys())
models_dict = get_models_dict()
pipe = None

# ===== State =====
gallery_images = []
caption_texts = []
character_dict = {}
processed_prompts = []
original_prompts = []
character_registry = {}  # persistent memory of characters and traits

# ===== Load Comic Model =====
model_name = "ComicModel"
model_info = models_dict[model_name]
pipe = load_models(model_info, device=device)
pipe.to(device)
pipe.enable_vae_slicing()
pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
if device != "mps":
    pipe.enable_model_cpu_offload()

# ===== Utils =====
def apply_style_positive(style_name: str, positive: str):
    p, _ = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive)

def setup_seed(seed):
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)

# ===== Character Registry =====
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

# ===== Main Generation =====
def process_generation(seed, style_name, general_prompt, prompt_array, font_choice,
                       steps, width, height, guidance_scale, comic_type):
    global gallery_images, caption_texts, original_prompts, character_dict, processed_prompts

    setup_seed(seed)
    character_dict = update_character_registry(general_prompt)

    prompts_raw = prompt_array.strip().splitlines()
    prompts_clean = [line.split("#")[0].replace("[NC]", "").strip() for line in prompts_raw]
    caption_texts = [line.split("#")[-1].strip() if "#" in line else "" for line in prompts_raw]

    _, _, processed_prompts, _, _ = process_original_prompt(character_dict, prompts_clean, 0)
    original_prompts = processed_prompts.copy()

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

    panel_choices = [str(i) for i in range(len(gallery_images))]
    return comic_images, gr.update(choices=panel_choices, value=panel_choices[0]), ""

# ===== Feedback Refinement Function =====
def refine_panel(index, refinement_text, style_name, steps, width, height, guidance_scale):
    global gallery_images, processed_prompts, character_dict, character_registry

    index = int(index)
    base_prompt = processed_prompts[index]
    character_tag = base_prompt.split("]")[0] + "]" if "]" in base_prompt else ""

    if refinement_text.strip():
        entry = character_registry.get(character_tag, {"base": "", "traits": []})
        if refinement_text.strip() not in entry["traits"]:
            entry["traits"].append(refinement_text.strip())
        character_registry[character_tag] = entry

    full_character = get_full_character_desc(character_tag)
    final_prompt = f"{character_tag} {full_character}. {base_prompt}"

    styled_prompt = apply_style_positive(style_name, final_prompt)

    setup_seed(random.randint(0, MAX_SEED))
    new_image = pipe(
        styled_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width
    ).images[0]

    gallery_images[index] = new_image
    return gallery_images

# ===== Add New Scene Mid-Story =====
def add_scene(new_scene, style_name, steps, width, height, guidance_scale):
    global gallery_images, processed_prompts, character_registry

    # Extract tag if exists
    character_tag = new_scene.split("]")[0] + "]" if "]" in new_scene else ""
    full_character = get_full_character_desc(character_tag)
    full_prompt = f"{character_tag} {full_character}. {new_scene}"

    styled_prompt = apply_style_positive(style_name, full_prompt)
    setup_seed(random.randint(0, MAX_SEED))
    new_image = pipe(
        styled_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width
    ).images[0]

    gallery_images.append(new_image)
    processed_prompts.append(new_scene)
    caption_texts.append("")  # Placeholder
    panel_choices = [str(i) for i in range(len(gallery_images))]
    return gallery_images, gr.update(choices=panel_choices, value=panel_choices[-1]), ""

# ===== Gradio UI =====
with gr.Blocks(title="NarrativeDiffusion with Character Memory & Scene Continuation") as demo:
    gr.Markdown("## 🎨 NarrativeDiffusion: Persistent Characters, Traits & Continuous Storytelling")

    with gr.Row():
        with gr.Column():
            general_prompt = gr.Textbox(label="🧍 Character Descriptions", lines=3,
                placeholder="[Tom] a boy with a red cape\n[Emma] a girl with silver hair")
            prompt_array = gr.Textbox(label="📜 Initial Story Prompts", lines=6,
                placeholder="[Tom] runs into the woods. #He looks around nervously.")
            style_dropdown = gr.Dropdown(choices=STYLE_NAMES, value=DEFAULT_STYLE_NAME, label="🎨 Art Style")
            font_choice = gr.Dropdown(choices=[f for f in os.listdir("fonts") if f.endswith(".ttf")],
                value="Inkfree.ttf", label="🖋️ Font")
            steps = gr.Slider(20, 50, value=30, step=1, label="🔁 Sampling Steps")
            guidance = gr.Slider(1.0, 10.0, value=5.0, step=0.5, label="🎯 Guidance Scale")
            width = gr.Slider(512, 1024, step=64, value=768, label="📐 Image Width")
            height = gr.Slider(512, 1024, step=64, value=768, label="📐 Image Height")
            comic_type = gr.Radio(["Classic Comic Style", "Four Pannel", "No typesetting (default)"],
                value="Classic Comic Style", label="🗂️ Layout")
            seed = gr.Slider(0, MAX_SEED, value=0, step=1, label="🌱 Seed")
            run_button = gr.Button("🚀 Generate Initial Story")

        with gr.Column():
            gallery = gr.Gallery(label="🖼️ Comic Panels", columns=2, height="auto")

            gr.Markdown("### 🔁 Refine Existing Panel")
            panel_selector = gr.Dropdown(choices=[], label="🎬 Select Panel to Refine")
            refine_prompt = gr.Textbox(label="✏️ Visual/Appearance Edit", placeholder="Tom now has a wound on his leg")
            refine_btn = gr.Button("🔧 Refine Panel")

            gr.Markdown("### ➕ Add a New Scene or Character Mid-Story")
            new_characters = gr.Textbox(label="👤 New Characters (optional)", placeholder="[Luna] a mysterious girl with glowing eyes")
            new_scene = gr.Textbox(label="📘 New Scene Prompt", placeholder="[Luna] enters the forest holding a glowing orb.")
            add_btn = gr.Button("➕ Add New Scene")

    run_button.click(
        fn=process_generation,
        inputs=[
            seed, style_dropdown, general_prompt, prompt_array, font_choice,
            steps, width, height, guidance, comic_type
        ],
        outputs=[gallery, panel_selector, refine_prompt]
    )

    refine_btn.click(
        fn=refine_panel,
        inputs=[panel_selector, refine_prompt, style_dropdown, steps, width, height, guidance],
        outputs=[gallery]
    )

    add_btn.click(
        fn=lambda new_chars, new_scene_text, *args: (
            update_character_registry(new_chars), add_scene(new_scene_text, *args)
        )[1],
        inputs=[new_characters, new_scene, style_dropdown, steps, width, height, guidance],
        outputs=[gallery, panel_selector, refine_prompt]
    )

demo.launch(share=True)
