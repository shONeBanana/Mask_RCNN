"""
Modified from matterport/Mask_RCNN/samples/shapes/
shapes.py
train_shapes.ipynb
"""

import os
import sys
import math
import random
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import json

DATASET_PATH = r"dataset\coco2014"
CLASS_NAME = r"cat"

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


class CustomDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_data(self, dataset_dir, subset):
        """Load a subset of the nuclei dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have one class.
        # Naming the dataset nucleus, and the class nucleus
        self.add_class(CLASS_NAME, 1, CLASS_NAME)

        # Which subset?
        # "val": use hard-coded list above
        # "train": use data from stage1_train minus the hard-coded list above
        # else: use the data from the specified sub-directory
        dataset_dir = os.path.join(dataset_dir, subset)

        # Get image ids from directory names
        file_list = [v for v in os.listdir(dataset_dir) if ".jpg" in v]
        image_ids = list(range(len(file_list)))
        print(image_ids)
        # Add images
        for image_id in image_ids:
            self.add_image("cat", image_id=image_id, path=os.path.join(dataset_dir, file_list[image_id]))

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        json_data = str(info['path']).replace(".jpg", ".json")
        fin = open(json_data, 'r')
        json_dict = json.load(fin)
        h, w = json_dict['imageHeight'], json_dict['imageWidth']
        masks = []
        for shape_data in json_dict["shapes"]:
            mask = np.zeros([h, w], np.uint8)
            points = np.array(shape_data["points"], np.int32)
            mask = cv2.fillPoly(mask, [points], 255)
            masks.append(mask)
        masks = np.stack(masks, axis=-1)
        fin.close()
        return masks, np.ones([masks.shape[-1]], dtype=np.int32)


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

if __name__ == '__main__':
    config = CustomConfig()
    config.display()

    # Training dataset
    dataset_train = CustomDataset()
    dataset_train.load_data(DATASET_PATH, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_data(DATASET_PATH, "val")
    dataset_val.prepare()

    # Load and display random samples
    image_ids = np.random.choice(dataset_train.image_ids, 4)
    for image_id in image_ids:
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

    #################
    # Train
    #################
    
    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
    
    # Which weights to start with?
    init_with = "coco"  # imagenet, coco, or last
    
    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)
    
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                layers='heads')
    
    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=2,
                layers="all")
    
    # End Train

    #################
    # Validation
    #################

    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    model_path = model.find_last()
    # Load trained weights
    print("# Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    # Test on a random image
    for _ in range(5):
        image_id = random.choice(dataset_val.image_ids)
        original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset_val, inference_config, image_id, use_mini_mask=False)

        log("original_image", original_image)
        log("image_meta", image_meta)
        log("gt_class_id", gt_class_id)
        log("gt_bbox", gt_bbox)
        log("gt_mask", gt_mask)

        # visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
        #                             dataset_train.class_names, figsize=(8, 8))

        results = model.detect([original_image], verbose=1)

        r = results[0]
        visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                dataset_val.class_names, r['scores'])

    # End Test

    #################
    # Calculate mAP
    #################

    # # Compute VOC-Style mAP @ IoU=0.5
    # # Running on 10 images. Increase for better accuracy.
    # image_ids = np.random.choice(dataset_val.image_ids, 10)
    # APs = []
    # for image_id in image_ids:
    #     # Load image and ground truth data
    #     image, image_meta, gt_class_id, gt_bbox, gt_mask = \
    #         modellib.load_image_gt(dataset_val, inference_config,
    #                                image_id, use_mini_mask=False)
    #     molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    #     # Run object detection
    #     results = model.detect([image], verbose=0)
    #     r = results[0]
    #     # Compute AP
    #     AP, precisions, recalls, overlaps = \
    #         utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
    #                          r["rois"], r["class_ids"], r["scores"], r['masks'])
    #     APs.append(AP)

    # print("mAP: ", np.mean(APs))
