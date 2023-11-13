 #coding=utf-8
import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
import argparse
import random
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import os
import torchvision
from torchvision.ops.boxes import batched_nms
import cv2
import matplotlib.pyplot as plt
import cv2

#两种表示方法之间的转换

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

#转换到实际的尺寸
def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

#对于阈值的box才留下来,否则将bbox剔除
def filter_boxes(scores, boxes, confidence=0.7, apply_nms=True, iou=0.5):
    keep = scores.max(-1).values > confidence
    # print(keep.shape)
    # print(keep)
    scores, boxes = scores[keep], boxes[keep]
    if apply_nms:
        top_scores, labels = scores.max(-1)
        keep = batched_nms(boxes, top_scores, labels, iou)
        scores, boxes = scores[keep], boxes[keep]
    return scores, boxes


def plot_results(img, prob, boxes):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img, aspect='equal')
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color='red', linewidth=2))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig('result.png')
    plt.draw()
    plt.show()
    



# img1=cv2.imread('./555_F0ed5ee7e728f4b94a22fe30e93b1009a.JPG')
# img2=cv2.rectangle(img1,(52,52),(200,200),(0,255,0),3)
# cv2.imwrite("./1.jpg", img2)

# plot_results(im, scores, boxes)

# class DETR(nn.Module):
#     def __init__(self, num_classes, hidden_dim, nheads,
#         num_encoder_layers, num_decoder_layers):
#         super().__init__()
# # We take only convolutional layers from ResNet-50 model
#         self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
#         self.conv = nn.Conv2d(2048, hidden_dim, 1)
#         self.transformer = nn.Transformer(hidden_dim, nheads,
#         num_encoder_layers, num_decoder_layers)
#         self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
#         self.linear_bbox = nn.Linear(hidden_dim, 4)
#         self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
#         self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
#         self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

#     def forward(self, inputs):
#         x = self.backbone(inputs)
#         h = self.conv(x)
#         H, W = h.shape[-2:]
#     #cat之后pos的尺寸是H*W*hidden_dim
#         pos = torch.cat([
#         self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
#         self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
#         ], dim=-1).flatten(0, 1).unsqueeze(1)
#         h = self.transformer(pos + h.flatten(2).permute(2, 0, 1),
#         self.query_pos.unsqueeze(1)).transpose(0, 1)
#         return self.linear_class(h), self.linear_bbox(h).sigmoid()

class DETR(nn.Module):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        # construct positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1)
        
        # finally project transformer outputs to class labels and bounding boxes
        return  self.linear_class(h), self.linear_bbox(h).sigmoid()
    


# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize((800,1200)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

from PIL import Image
import numpy as np
# from DETR_detect.tools.function import *
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

img_path= "./test.png"
image = Image.open(img_path)
# source_image=image
source_image=transform(image).unsqueeze(0)
print(source_image.shape)
# print(source_image.shape)

# source_image = np.expand_dims(np.array(source_image).transpose((2,0,1)),0)
# source_image=torch.from_numpy((source_image)).float()

detr = DETR(num_classes=91, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6)
state_dict = torch.load("detr.pth")
detr.load_state_dict(state_dict)
detr.eval()
logits, bboxes = detr(source_image)

#通过这两个返回的参数计算loss
# print(logits.shape)
pre_class = logits.softmax(-1)[0, :, :-1].cpu()
# pre_class = pre_class[0, :, :-1].cpu()
# print(pre_class.shape)
bboxes_scaled = rescale_bboxes(bboxes[0,].cpu(), (source_image.shape[3], source_image.shape[2]))
print("hello,world")

score, pre_box = filter_boxes(pre_class, bboxes_scaled)
# print(score.shape)
plot_results(image, score, pre_box)

# class_id = score.argmax()
# print(class_id)
# label = CLASSES[class_id] 
# confidence = score.max().float()
# text = f'{label} {confidence:.3f}'
# # text=label+str(confidence)
# print(text)

# state_dict = torch.hub.load_state_dict_from_url(
#     url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
#     map_location='cpu', check_hash=True)