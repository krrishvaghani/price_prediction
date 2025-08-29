from PIL import Image

def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    image = image.convert("RGB")
    return image