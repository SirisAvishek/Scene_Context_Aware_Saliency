import torch
from torch import nn
from torch.nn import functional as F

from detectron2.structures import ImageList

from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from SaliencyROIHeads import build_saliency_roi_heads
from Context_Segmentation_FPN_Decoder import build_context_seg_decoder
from Context_Segmentation_FPN_Head import build_context_seg_head

from Scene_Context_Refinement_Module import build_scene_context_refinement_module

# BASELINE_MODEL_REGISTRY = Registry("BASELINE_MODEL_REGISTRY")
__all__ = ["SC_Model"]


@META_ARCH_REGISTRY.register()
class SC_Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.instance_loss_weight = cfg.MODEL.PANOPTIC_FPN.INSTANCE_LOSS_WEIGHT

        # options when combining instance & semantic outputs
        self.combine_on = cfg.MODEL.PANOPTIC_FPN.COMBINE.ENABLED
        self.combine_overlap_threshold = cfg.MODEL.PANOPTIC_FPN.COMBINE.OVERLAP_THRESH
        self.combine_stuff_area_limit = cfg.MODEL.PANOPTIC_FPN.COMBINE.STUFF_AREA_LIMIT
        self.combine_instances_confidence_threshold = (
            cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH  # 0.5
        )

        self.backbone = build_backbone(cfg)

        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())

        self.roi_heads = build_saliency_roi_heads(cfg, self.backbone.output_shape())

        # --------------------------------------------------

        self.context_seg_decoder = build_context_seg_decoder(cfg, self.backbone.output_shape())
        self.context_seg_head = build_context_seg_head(cfg)

        # --------------------------------------------------

        self.scene_context_refinement_module = build_scene_context_refinement_module(cfg)

        # --------------------------------------------------

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.

                For now, each item in the list is a dict that contains:

                * "image": Tensor, image in (C, H, W) format.
                * "instances": Instances
                * "sem_seg": semantic segmentation ground truth.
                * Other information that's included in the original dicts, such as:
                  "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                each dict is the results for one image. The dict contains the following keys:

                * "instances": see :meth:`GeneralizedRCNN.forward` for its format.
                * "sem_seg": see :meth:`SemanticSegmentor.forward` for its format.
                * "panoptic_seg": available when `PANOPTIC_FPN.COMBINE.ENABLED`.
                  See the return value of
                  :func:`combine_semantic_and_instance_outputs` for its format.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        # --------------------------------------------------

        if "proposals" in batched_inputs[0]:
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        # self.backbone.size_divisibility = 32
        ignore_value = 255
        if "inst_context_seg" in batched_inputs[0]:
            gt_inst_context_seg = [x["inst_context_seg"].to(self.device) for x in batched_inputs]
            gt_inst_context_seg = ImageList.from_tensors(
                gt_inst_context_seg, self.backbone.size_divisibility, ignore_value).tensor
        else:
            gt_inst_context_seg = None
        if "stuff_context_seg" in batched_inputs[0]:
            gt_stuff_context_seg = [x["stuff_context_seg"].to(self.device) for x in batched_inputs]
            gt_stuff_context_seg = ImageList.from_tensors(
                gt_stuff_context_seg, self.backbone.size_divisibility, ignore_value).tensor
        else:
            gt_stuff_context_seg = None

        # --------------------------------------------------

        # Context Segmentation Decoder
        context_seg_features = self.context_seg_decoder(features)

        # Context Segmentation Heads - Instance/Stuff
        inst_context_seg_results, stuff_context_seg_results, context_seg_losses = \
            self.context_seg_head(context_seg_features, gt_inst_context_seg, gt_stuff_context_seg)

        # --------------------------------------------------

        refined_context_feat = self.scene_context_refinement_module(features, context_seg_features)

        # --------------------------------------------------

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)

        instance_results, detector_losses = self.roi_heads(images, features, proposals, refined_context_feat,
                                                           gt_instances)

        # --------------------------------------------------
        # Training
        if self.training:
            losses = {}
            losses.update(context_seg_losses)
            losses.update({k: v * self.instance_loss_weight for k, v in detector_losses.items()})
            losses.update(proposal_losses)
            return losses

        # --------------------------------------------------
        # Inference
        processed_results = generate_instance_saliency_map(instance_results,
                                                           inst_context_seg_results, stuff_context_seg_results,
                                                           batched_inputs, images,
                                                           context_seg_features, refined_context_feat,
                                                           self.combine_overlap_threshold,
                                                           self.combine_instances_confidence_threshold)

        return processed_results


def generate_instance_saliency_map(instance_predictions, inst_context_seg_prediction, stuff_context_seg_prediction,
                                   batched_inputs, images,
                                   context_seg_ft, refined_context_ft,
                                   overlap_threshold, instances_confidence_threshold):
    results = []
    for instance_result, inst_context_seg_result, stuff_context_seg_result, \
        input_per_image, image_size, \
        con_seg_ft, refined_con_ft in zip(instance_predictions,
                                          inst_context_seg_prediction, stuff_context_seg_prediction,
                                          batched_inputs, images.image_sizes,
                                          context_seg_ft, refined_context_ft):
        height = input_per_image.get("height")
        width = input_per_image.get("width")

        detector_r = detector_postprocess(instance_result, height, width)

        inst_seg_r = seg_postprocess(inst_context_seg_result, image_size, height, width)
        stuff_seg_r = seg_postprocess(stuff_context_seg_result, image_size, height, width)

        instance_map = generate_instance_output_map(detector_r,
                                                    inst_seg_r.argmax(dim=0),
                                                    overlap_threshold, instances_confidence_threshold)

        results.append({"sal_map": instance_map, "inst_con_seg": inst_seg_r,
                        "stuff_con_seg": stuff_seg_r,
                        "con_seg_ft": con_seg_ft, "refined_con_ft": refined_con_ft})

    return results


def generate_instance_output_map(instance_results, segmentation_results,
                                 overlap_threshold, instances_confidence_threshold):
    instance_seg = torch.zeros_like(segmentation_results, dtype=torch.float32)

    # sort instance outputs by scores
    sorted_inds = torch.argsort(-instance_results.scores)

    instance_masks = instance_results.pred_masks.to(dtype=torch.bool, device=instance_seg.device)

    # Add instances one-by-one, check for overlaps with existing ones
    for inst_id in sorted_inds:
        score = instance_results.scores[inst_id].item()
        if score < instances_confidence_threshold:
            break
        mask = instance_masks[inst_id]  # H,W
        mask_area = mask.sum().item()

        if mask_area == 0:
            continue

        intersect = (mask > 0) & (instance_seg > 0)
        intersect_area = intersect.sum().item()

        if intersect_area * 1.0 / mask_area > overlap_threshold:
            continue

        if intersect_area > 0:
            mask = mask & (instance_seg == 0)

        if instance_results.pred_classes[inst_id].item() == 1:
            instance_seg[mask] = 255

    return instance_seg


def seg_postprocess(result, img_size, output_height, output_width):
    # Resize to original size
    result = result[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)

    result = F.interpolate(
        result, size=(output_height, output_width), mode="bilinear", align_corners=False
    )[0]

    return result
