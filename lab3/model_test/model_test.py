from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt


# Function to generate ADE20K colormap
def ade20k_colormap():
    cmap = plt.get_cmap("tab20", 150)
    return (np.array([cmap(i)[:3] for i in range(150)]) * 255).astype(np.uint8)


# Load the model and image processor
model_name = "facebook/mask2former-swin-large-ade-semantic"
# model_name = "../model"
image_processor = AutoImageProcessor.from_pretrained(model_name)
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)

# Load and preprocess the image
image = Image.open("test.jpg").convert("RGB")
image = image.resize((1024, 512))

# Call the model
inputs = image_processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)


# class_queries_logits = outputs.class_queries_logits
# mask_queries_logits = outputs.masks_queries_logits

# Get the predicted masks and logits
predicted = image_processor.post_process_semantic_segmentation(
    outputs, target_sizes=[image.size[::-1]]
)[0]


# Visualize the segmentation map
alpha = 0.9
segmentation_map = predicted.numpy()
color_seg = ade20k_colormap()[segmentation_map]
mask = segmentation_map != 0
original_np = np.array(image)
blended = np.where(
    mask[..., None], (1 - alpha) * original_np + alpha * color_seg, original_np
).astype(np.uint8)

# Save result
blended_img = Image.fromarray(blended)
blended_img.save("segmentation_overlay.png")


# Use to save the model and image processor locally
# image_processor.save_pretrained("../model")
# model.save_pretrained("../model")