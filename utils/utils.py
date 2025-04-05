from PIL import Image, ImageDraw, ImageFont

def get_comic(images, layout_type="Classic Comic Style", captions=None, font=None):
    padded_images = []
    for i, img in enumerate(images):
        draw = ImageDraw.Draw(img)
        if captions and i < len(captions):
            draw.text((20, 20), captions[i], font=font, fill=(0, 0, 0))
        padded_images.append(img)

    return padded_images
