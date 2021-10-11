import os
import utils

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.logger import setup_logger

from SaliencyTrainer import SaliencyTrainer

# Import Model, so it is added to the model registry
from SC_Model import SC_Model

from torch import autograd

import torch
torch.cuda.empty_cache()

setup_logger()


def register_dataset(name, metadata, image_root, gt_image_root, inst_context_image_root, stuff_context_image_root,
                     instances_json):
    DatasetCatalog.register(
        name,
        lambda: utils.get_tsc_instances_dicts(image_root, gt_image_root,
                                              inst_context_image_root, stuff_context_image_root, instances_json)
    )
    MetadataCatalog.get(name).set(
        image_root=image_root,
        gt_image_root=gt_image_root,
        inst_context_image_root=inst_context_image_root,
        stuff_context_image_root=stuff_context_image_root,
        instances_json=instances_json,
        thing_classes=["salient", "context"],
        **metadata
    )

    print("Dataset Registered!!!")


if __name__ == "__main__":
    name = "coco_train_TSC"
    metadata = {}

    model_name = "SC_Model"

    dataset_root = "D:/Desktop/SCAS_Dataset/"

    image_root = dataset_root + "train/images/"
    gt_image_root = dataset_root + "train/saliency_maps/"
    inst_context_image_root = dataset_root + "train/instance_context_maps/"
    stuff_context_image_root = dataset_root + "train/stuff_context_maps/"
    instances_json = dataset_root + "TSC_instances_train.json"

    register_dataset(name, metadata, image_root, gt_image_root,
                     inst_context_image_root, stuff_context_image_root, instances_json)

    dataset_metadata = MetadataCatalog.get(name)

    # --------------------------------------------------
    num_images = len([f for f in os.listdir(image_root)])
    print("No. of Images: ", num_images)

    # --------------------------------------------------

    epoch = 30
    cfg = get_cfg()
    cfg.merge_from_file('models/sc_model.yaml')
    cfg.DATASETS.TRAIN = ('coco_train_TSC',)  # the comma is necessary
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.OUTPUT_DIR = 'trained_model/{}_epoch_{}'.format(model_name, epoch)

    cfg.MODEL.WEIGHTS = 'weights/sc_model_init_weights.pth'

    cfg.SOLVER.MAX_ITER = num_images * epoch

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = SaliencyTrainer(cfg)

    trainer.resume_or_load(resume=False)

    with autograd.detect_anomaly():
        trainer.train()
