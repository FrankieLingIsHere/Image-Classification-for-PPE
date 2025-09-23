#!/usr/bin/env python3
"""
Simple PPE Detection Test Script
"""

import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.ssd import SSD300

def simple_test(model_path, image_path):
    """Simple model test"""
    
    print("Testing PPE Detection Model")
    print("=" * 40)
    
    # Check if files exist
    if not os.path.exists(model_path):
        print("Model file not found:", model_path)
        return
    
    if not os.path.exists(image_path):
        print("Image file not found:", image_path)
        return
    
    # Load model
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device)
        print("Checkpoint loaded successfully")
        print("Keys in checkpoint:", list(checkpoint.keys()))
        
        # Check if training info exists
        if 'epoch' in checkpoint:
            print("Trained for epochs:", checkpoint['epoch'])
        if 'loss' in checkpoint:
            print("Final loss:", checkpoint['loss'])
            
    except Exception as e:
        print("Error loading checkpoint:", e)
        return
    
    # Load and display image info
    print("\nImage Information:")
    image = cv2.imread(image_path)
    if image is not None:
        h, w, c = image.shape
        print("Image size:", w, "x", h)
        print("Channels:", c)
    else:
        print("Could not load image")
        return
    
    print("\nModel testing setup complete!")
    print("For full inference, use: python scripts/test_model.py")

def main():
    parser = argparse.ArgumentParser(description='Simple PPE Model Test')
    parser.add_argument('--model', type=str, default='models/checkpoint_epoch_4.pth')
    parser.add_argument('--image', type=str, default='data/images')
    
    args = parser.parse_args()
    
    # If image is a directory, find first image
    if os.path.isdir(args.image):
        image_files = [f for f in os.listdir(args.image) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if image_files:
            args.image = os.path.join(args.image, image_files[0])
        else:
            print("No images found in directory:", args.image)
            return
    
    simple_test(args.model, args.image)

if __name__ == "__main__":
    main()