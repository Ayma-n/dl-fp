import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def get_image_encoding(filepath: str):
    image = preprocess(Image.open(filepath)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        return image_features.numpy()

def get_text_encoding(tokens: list[str]):
    text = clip.tokenize(tokens).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        return text_features