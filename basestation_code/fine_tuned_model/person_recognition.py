import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Load model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load and preprocess image
image = Image.open("path_to_your_image.jpg")

# Text prompts
texts = ["a photo of a human", "no humans in this image"]

# Prepare inputs
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

# Get embeddings
outputs = model(**inputs)
image_embeds = outputs.image_embeds  # (1, dim)
text_embeds = outputs.text_embeds    # (2, dim)

# Normalize embeddings
image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

# Compute cosine similarity between image and each text
cosine_similarities = (image_embeds @ text_embeds.T).squeeze()

print(f"Similarity to 'human': {cosine_similarities[0].item():.4f}")
print(f"Similarity to 'no humans': {cosine_similarities[1].item():.4f}")

# Decision threshold (you can adjust)
if cosine_similarities[0] > cosine_similarities[1]:
    print("Human detected in the image!")
else:
    print("No human detected.")
