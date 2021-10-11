# Scene Context-Aware Salient Object Detection [ICCV 2021]

### Authors:
Avishek Siris, Jianbo Jiao, Gary K.L. Tam, Xianghua Xie, Rynson W.H. Lau

+ PDF: [Paper]
+ Supplemental: [Supplementary Material]


##
<p align="center">
<img src="https://github.com/SirisAvishek/Scene_Context_Aware_Saliency/blob/main/model_overview.jpg" width="800"/>
</p>

## Abstract
Salient object detection identifies objects in an image that grab visual attention. Recently, many deep models are proposed to capture contextual features to improve salient object detection. These methods, however, often fail when the input images contain complex scenes and backgrounds. We observe that this failure is mainly caused by two problems. First, most existing datasets consist of largely simple foregrounds and backgrounds that hardly represent real-life scenarios. Second, current methods only learn contextual features of salient object, which is insufficient to model high-level semantics for saliency reasoning in complex scenes. To address these problems, in this paper, a new large-scale dataset with complex scenes is first constructed. We then propose a context-aware learning approach to explicitly exploit the semantic scene contexts. Specifically, two modules are proposed to achieve the goal: 1) a Semantic Scene Context Refinement module to enhance contextual features learned from salient objects with scene context, and 2) a Contextual Instance Transformer to learn contextual relations between objects and scene context. To our knowledge, such high-level context information of image scenes is under-explored for saliency detection in the literature. Extensive experiments demonstrate that the proposed approach outperforms state-of-the-art techniques in complex scenarios for saliency detection, and transfers well to other existing datasets (see supplementary).

## Installation
The code is based on the [detectron2](https://github.com/facebookresearch/detectron2) framework. Please follow and install the requirements they list.

## Dataset
Download our dataset from [google drive](https://drive.google.com/file/d/1x7y-mzFZhIKrLsL-CPNs4xrhanWxW030/view?usp=sharing).

## Training 
Download our pre-trained weights for initilisation from [google drive](https://drive.google.com/file/d/1vLbX6dOj_XHw2RfuKgqKA54feRvsRwiE/view?usp=sharing). Create a new "weights/" folder in the root directory and place the weights file inside it.
Set data paths and run:
```
python Train_SC_Model.py 
```

## Testing
You can download the weights of the trained model from [google drive](https://drive.google.com/file/d/1jpLDVIdwP5gjO0RQlm81Zlhg21u515xj/view?usp=sharing).
Set data paths and run:
```
python Test_SC_Model.py
```

## Results
You can download predicted Saliency Maps by our trained model from [google drive](https://drive.google.com/file/d/106bk6X5NYVCorbu91MQ8z0KjkWj8MkVz/view?usp=sharing).

# Citation
```
@InProceedings{Siris_2021_ICCV,
author = {Siris, Avishek and Jiao, Jianbo and Tam, Gary K.L. and Xie, Xianghua and Lau, Rynson W.H.},
title = {Scene Context-Aware Salient Object Detection},
booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
month = {October},
year = {2021}
}
```


