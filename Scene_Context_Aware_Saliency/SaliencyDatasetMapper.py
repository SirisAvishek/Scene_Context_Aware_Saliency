import copy
import logging
import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    PolygonMasks,
)


class SaliencyDatasetMapper:

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None

        self.tfm_gens = utils.build_transform_gen(cfg, is_train)

        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = cfg.MODEL.MASK_ON
        self.mask_format    = cfg.INPUT.MASK_FORMAT
        self.keypoint_on    = False
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        # fmt: on

        self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train

    def __call__(self, dataset_dict):

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        # Can use uint8 if it turns out to be slow some day

        # USER: Remove if you don't use pre-computed proposals.
        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, self.min_box_side_len, self.proposal_topk
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]

            instances = annotations_to_instances(annos, image_shape, mask_format=self.mask_format)

            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        if "inst_context_file_name" in dataset_dict:
            with PathManager.open(dataset_dict.pop("inst_context_file_name"), "rb") as f:
                inst_con_seg_gt = Image.open(f)
                inst_con_seg_gt = np.asarray(inst_con_seg_gt, dtype="uint8")
            inst_con_seg_gt = transforms.apply_segmentation(inst_con_seg_gt)

            # Required for ImageList.from_tensors in Model
            # Image seems to open as (H, W, 3), when (H, W) is required
            # Reason likely because of the method for generating the map
            inst_con_seg_gt = inst_con_seg_gt[:, :, 0]

            inst_con_seg_gt = torch.as_tensor(inst_con_seg_gt.astype("long"))
            dataset_dict["inst_context_seg"] = inst_con_seg_gt

        if "stuff_context_file_name" in dataset_dict:
            with PathManager.open(dataset_dict.pop("stuff_context_file_name"), "rb") as f:
                stuff_con_seg_gt = Image.open(f)
                stuff_con_seg_gt = np.asarray(stuff_con_seg_gt, dtype="uint8")
            stuff_con_seg_gt = transforms.apply_segmentation(stuff_con_seg_gt)

            stuff_con_seg_gt = torch.as_tensor(stuff_con_seg_gt.astype("long"))
            dataset_dict["stuff_context_seg"] = stuff_con_seg_gt

        return dataset_dict


def annotations_to_instances(annos, image_size, mask_format="polygon"):
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    boxes = target.gt_boxes = Boxes(boxes)
    boxes.clip(image_size)

    '''
        "category_id" has been swapped with "sal_cls_id" during the loading and processing 
        of the dataset in the utils function
    '''

    classes = [obj["category_id"] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    if len(annos) and "segmentation" in annos[0]:
        polygons = [obj["segmentation"] for obj in annos]
        if mask_format == "polygon":
            masks = PolygonMasks(polygons)
        else:
            assert mask_format == "bitmask", mask_format
            masks = BitMasks.from_polygon_masks(polygons, *image_size)
        target.gt_masks = masks

    return target











