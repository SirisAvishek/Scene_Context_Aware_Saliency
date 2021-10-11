import os
import cv2
import numpy as np

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

# Import Model, so it is added to the model registry
from SC_Model import SC_Model


def cv2_imshow(img_array):
    cv2.imshow("img", img_array)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    dataset_root = "D:/Desktop/SCAS_Dataset/"

    weight_file_name = "sc_model_weights.pth"

    # ----------------------------------------------------------------------------------------------

    # Inference with model
    cfg = get_cfg()
    cfg.merge_from_file("models/sc_model.yaml")
    cfg.MODEL.WEIGHTS = "weights/" + weight_file_name

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 512
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    predictor = DefaultPredictor(cfg)

    # Output Location
    output_dir = "saliency_prediction/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Test Images Location
    test_images_dir = dataset_root + "val/images/"
    test_images = [f for f in os.listdir(test_images_dir)]
    print("Test Images: ", len(test_images))

    num = len(test_images)
    for i in range(num):
        print("\n", i+1, " / ", num)

        img = test_images[i]

        im_path = test_images_dir + img
        im = cv2.imread(im_path)

        predictions = predictor(im)

        pred_sal_map = predictions["sal_map"]
        pred_sal_map = pred_sal_map.cpu().data.numpy()

        # ---------- Save Prediction Image
        out_path = output_dir + img[:-3] + "png"

        sal_mask = np.zeros(shape=(480, 640, 3), dtype=np.float32)
        sal_mask[:, :, 0] = pred_sal_map
        sal_mask[:, :, 1] = pred_sal_map
        sal_mask[:, :, 2] = pred_sal_map
        sal_mask *= 255

        cv2.imwrite(out_path, sal_mask)

