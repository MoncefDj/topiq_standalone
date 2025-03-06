import os
import argparse
import torch
import numpy as np
from PIL import Image
import time
from pathlib import Path

from topiq_model import create_topiq, load_image

def process_image(model, image_path, ref_image_path=None, device='cpu'):
    """
    Process a single image with the TOPIQ model
    
    Args:
        model: The TOPIQ model
        image_path: Path to the image to assess
        ref_image_path: Path to reference image (for FR models)
        device: Device to run inference on
        
    Returns:
        Quality score
    """
    # Load and preprocess image
    img = load_image(image_path).to(device)
    
    # Process reference image if provided
    ref_img = None
    if ref_image_path is not None:
        ref_img = load_image(ref_image_path).to(device)
    
    # Get quality score
    with torch.no_grad():
        score = model(img, ref_img)
    
    return score.item()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="TOPIQ Image Quality Assessment")
    parser.add_argument("--image", "-i", type=str, required=True, 
                        help="Path to the image to assess")
    parser.add_argument("--reference", "-r", type=str, default=None,
                        help="Path to reference image (for FR models)")
    parser.add_argument("--model", "-m", type=str, default="topiq_nr",
                        choices=["topiq_nr", "topiq_nr-face", "topiq_fr", "topiq_iaa"],
                        help="TOPIQ model variant to use")
    parser.add_argument("--device", "-d", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on (cuda or cpu)")
    parser.add_argument("--weights", "-w", type=str, default=None,
                        help="Path to custom model weights")
    args = parser.parse_args()
    
    # Check if the image exists
    if not os.path.exists(args.image):
        print(f"Error: Image {args.image} does not exist")
        return
    
    # Check if reference image exists if provided
    if args.reference is not None and not os.path.exists(args.reference):
        print(f"Error: Reference image {args.reference} does not exist")
        return
    
    # Create cache directory for model weights
    cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'topiq')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create the model
    print(f"Loading {args.model} model...")
    model = create_topiq(
        model_name=args.model,
        device=args.device,
        pretrained=True,
        pretrained_model_path=args.weights
    )
    
    # Process the image
    start_time = time.time()
    score = process_image(model, args.image, args.reference, args.device)
    end_time = time.time()
    
    # Print results
    print(f"\nImage: {args.image}")
    if args.reference:
        print(f"Reference: {args.reference}")
    print(f"Model: {args.model}")
    print(f"Quality Score: {score:.4f}")
    print(f"Inference Time: {(end_time - start_time)*1000:.2f} ms")
    
    # Interpret the score
    if args.model.startswith('topiq_iaa'):
        print(f"Aesthetic Quality (1-10): {score:.2f}")
    else:
        print(f"Technical Quality (0-1): {score:.4f}")

if __name__ == "__main__":
    main()
