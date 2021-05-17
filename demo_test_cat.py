"""
Modified from matterport/Mask_RCNN/samples/shapes/
shapes.py
train_shapes.ipynb
"""

import os
import sys
import skimage.io
# import math
# import random
# import numpy as np
# import cv2
# import matplotlib
# import matplotlib.pyplot as plt
# import json

DATASET_PATH = r"dataset\coco2014\test"
CLASS_NAME = r"cat"
MODEL_PATH = r"logs\cat20210517T1437\mask_rcnn_cat_0002.h5"

# Root directory of the project
ROOT_DIR = os.path.abspath("")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class CustomConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = CLASS_NAME

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4  # 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1   # background + cat

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 10

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


class InferenceConfig(CustomConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


if __name__ == '__main__':
    config = InferenceConfig()
    config.display()
    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(MODEL_PATH, by_name=True)

    class_names = ['BG', 'cat']

    # Load a random image from the images folder
    file_names = next(os.walk(DATASET_PATH))[2]
    for file_name in file_names[:3]:
        image = skimage.io.imread(os.path.join(DATASET_PATH, file_name))

        # Run detection
        results = model.detect([image], verbose=1)

        # Visualize results
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])