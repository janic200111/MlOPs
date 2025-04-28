import boto3
import os
import tempfile
from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt


# Function to generate ADE20K colormap
def ade20k_colormap():
    cmap = plt.get_cmap("tab20", 150)
    return (np.array([cmap(i)[:3] for i in range(150)]) * 255).astype(np.uint8)


# Lambda handler
def lambda_handler(event, context):

    # Initialize S3 client
    s3 = boto3.client("s3")

    # Load model and processor once
    # model_name = "facebook/mask2former-swin-large-ade-semantic"
    local_model_path = "/var/task/model"
    image_processor = AutoImageProcessor.from_pretrained(local_model_path)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(local_model_path)

    # Parse event
    source_bucket = event["Records"][0]["s3"]["bucket"]["name"]
    source_key = event["Records"][0]["s3"]["object"]["key"]

    # Define target bucket
    target_bucket = "staglo-output-bucket"

    # Temporary file paths
    with tempfile.TemporaryDirectory() as tmpdir:

        # Check if the uploaded file is a jpg or jpeg
        if not source_key.lower().endswith((".jpg", ".jpeg")):
            return {
                "statusCode": 400,
                "body": "Uploaded file is not a valid JPG or JPEG image.",
            }

        # Create temporary file paths
        download_path = os.path.join(tmpdir, "input.jpg")
        upload_path = os.path.join(
            tmpdir, f"{os.path.splitext(os.path.basename(source_key))[0]}.png"
        )

        # Download image from S3
        s3.download_file(source_bucket, source_key, download_path)

        # Load and preprocess image
        image = Image.open(download_path).convert("RGB")
        image = image.resize((1024, 512))

        # Inference
        inputs = image_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process segmentation
        predicted = image_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]

        # Blend segmentation overlay
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
        blended_img.save(upload_path)

        # Test without a model
        # image.save(upload_path)

        # Upload processed -image to target bucket
        target_key = f"processed/{os.path.basename(source_key)}"
        s3.upload_file(upload_path, target_bucket, target_key)

    return {
        "statusCode": 200,
        "body": f"Image processed and saved to {target_bucket}/{target_key}",
    }
