def character_to_dict(description_text):
    character_dict = {}
    lines = description_text.strip().split("\n")
    for line in lines:
        if "]" in line:
            name = line.split("]")[0][1:]
            desc = line.split("]")[1].strip()
            character_dict[name] = desc
    return character_dict, list(character_dict.keys())


def process_original_prompt(character_dict, prompts, id_length):
    character_index_dict = {}
    invert_character_index_dict = {}
    replace_prompts = []
    ref_indexs_dict = {}
    ref_totals = []

    for i, prompt in enumerate(prompts):
        for char in character_dict.keys():
            if f"[{char}]" in prompt:
                character_index_dict[char] = i
                invert_character_index_dict[i] = char
                prompt = prompt.replace(f"[{char}]", character_dict[char])
                ref_totals.append(i)
                if char not in ref_indexs_dict:
                    ref_indexs_dict[char] = []
                ref_indexs_dict[char].append(i)
        replace_prompts.append(prompt)

    return character_index_dict, invert_character_index_dict, replace_prompts, ref_indexs_dict, ref_totals


def get_ref_character(prompt, character_dict):
    characters = []
    for char in character_dict:
        if f"[{char}]" in prompt:
            characters.append(char)
    return characters


def is_torch2_available():
    try:
        import torch
        return int(torch.__version__.split(".")[0]) >= 2
    except:
        return False


class AttnProcessor:
    def __init__(self):
        pass
