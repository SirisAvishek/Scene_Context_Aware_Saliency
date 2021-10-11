import json
import logging

import cv2
from detectron2.structures import BoxMode
from pycocotools import mask as maskUtils

logger = logging.getLogger(__name__)


def get_obj_mask(seg_ann_data, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    if isinstance(seg_ann_data, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(seg_ann_data, height, width)
        rle = maskUtils.merge(rles)
    elif isinstance(seg_ann_data['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(seg_ann_data, height, width)
    else:
        rle = seg_ann_data

    m = maskUtils.decode(rle)

    return m


def get_tsc_instances_dicts(image_root, gt_image_root, inst_context_image_root, stuff_context_image_root, json_path):
    print("\n")
    print("Loading Instances Data ...")

    with open(json_path) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "category_id", "sal_cls_id"]

    num_instances_without_valid_segmentation = 0

    num = len(imgs_anns)
    for k in range(num):
        im_ann = imgs_anns[k]

        im_path = image_root + im_ann["image_id"] + ".jpg"
        height, width = cv2.imread(im_path).shape[:2]

        gt_path = gt_image_root + im_ann["image_id"] + ".png"
        inst_con_path = inst_context_image_root + im_ann["image_id"] + ".png"
        stuff_con_path = stuff_context_image_root + im_ann["image_id"] + ".png"

        # -----

        objs = []

        obj_ann = imgs_anns[k]["obj_annotation"]

        for anno in obj_ann:
            obj = {key: anno[key] for key in ann_keys if key in anno}

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if not isinstance(segm, dict):
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            obj["bbox_mode"] = BoxMode.XYWH_ABS

            # Swap "category_id" with "sal_cls_id"
            sal_id = anno.get("sal_cls_id", None)
            if sal_id:
                obj["category_id"] = sal_id

                # Non-salient objects are 0 in the annotation data
                # But we need 0 for BG class
                if sal_id == 0:
                    obj["category_id"] = 2
            else:
                obj["category_id"] = 0

            objs.append(obj)

        # ----------------------------------------

        record = {}
        record["file_name"] = im_path
        record["image_id"] = im_ann["image_id"]
        record["height"] = height
        record["width"] = width

        record["gt_file_name"] = gt_path
        record["inst_context_file_name"] = inst_con_path
        record["stuff_context_file_name"] = stuff_con_path

        record["annotations"] = objs

        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warn(
            "Filtered out {} instances without valid segmentation. "
            "There might be issues in your dataset generation process.".format(
                num_instances_without_valid_segmentation
            )
        )

    print("Loading Instances Data Complete!!!")

    return dataset_dicts

