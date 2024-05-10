import torch
import clip
from PIL import Image
import tensorflow as tf

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def get_image_encoding_from_filepath(filepath: str):
    image = preprocess(Image.open(filepath)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        return image_features.cpu().numpy()

def get_tokens(tokens: list[str]):
    text = clip.tokenize(tokens).to(device)
    return text.cpu().numpy()

def get_text_encoding(tokens: list[str]):
    text = clip.tokenize(tokens).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        return text_features.cpu().numpy()
    
def batch_get_image_encodings(images: tf.Tensor):
    torch_tensor = torch.from_numpy(images.cpu().numpy()).permute(0, 3, 1, 2).to(device) # Change from [N, H, W, C] to [N, C, H, W] (different convention)
    with torch.no_grad():
        image_features = model.encode_image(torch_tensor)
        return image_features.cpu().numpy()