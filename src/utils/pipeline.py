import os
import numpy as np
import tensorflow as tf
import cv2
import albumentations as A
import mlflow
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

PALETTE_8CLASS = np.array([
    [128, 64, 128],   # 0: route
    [244, 35, 232],   # 1: trottoir
    [70, 70, 70],     # 2: bâtiments/murs
    [220, 220, 0],    # 3: panneaux/feux
    [107, 142, 35],   # 4: végétation/terrain
    [70, 130, 180],   # 5: ciel
    [220, 20, 60],    # 6: piétons/riders
    [0, 0, 142],      # 7: véhicules
    [0, 0, 0]         # 255: ignore (affiché en noir)
], dtype=np.uint8)

def parse_example(example_proto):
    """
    Décode un exemple TFRecord en couple (image, masque).
    """
    # Description du format
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "mask": tf.io.FixedLenFeature([], tf.string),
    }

    # Parsing
    example = tf.io.parse_single_example(example_proto, feature_description)

    # Décodage
    image = tf.io.decode_png(example["image"], channels=3)  # (H, W, 3)
    mask = tf.io.decode_png(example["mask"], channels=1)    # (H, W, 1)

    return image, mask

def decode_segmentation_mask(mask, palette=PALETTE_8CLASS):
    """
    Convertit une image de masques (valeurs entre 0 et 7 ou 255) en image RGB.
    """
    rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)

    for class_id in range(len(palette)):
        rgb_mask[mask == class_id] = palette[class_id]

    return rgb_mask

def show_image_mask_batch_colored(dataset, n=3):
    images, masks = [], []

    for img, mask in dataset.take(n):
        images.append(img.numpy())
        masks.append(decode_segmentation_mask(mask.numpy()[..., 0]))

    plt.figure(figsize=(4 * n, 6))
    
    for i in range(n):
        plt.subplot(2, n, i + 1)
        plt.imshow(images[i])
        plt.title(f"Image {i+1}")
        plt.axis("off")
        
    for i in range(n):
        plt.subplot(2, n, n + i + 1)
        plt.imshow(masks[i])
        plt.title(f"Masque {i+1}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def normalize_image_mask(image, mask):
    """
    Normalise l’image entre 0 et 1. Laisse le masque tel quel.
    """
    image = tf.cast(image, tf.float32) / 255.0  # Normalisation
    return image, mask