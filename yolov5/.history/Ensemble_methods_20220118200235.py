import gc
import torchvision
import torch
import numpy as np
import pandas as pd
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import matplotlib.pyplot as plt
from ensemble_boxes import *
import cv2





marking = pd.read_csv('./text_results.csv')

# bboxs = np.stack(marking['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
# for i, column in enumerate(['x', 'y', 'w', 'h']):
#     marking[column] = bboxs[:,i]
# marking.drop(columns=['bbox'], inplace=True)


print(marking.head())



# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# df_folds = marking[['image_id']].copy()
# df_folds.loc[:, 'bbox_count'] = 1
# df_folds = df_folds.groupby('image_id').count()
# df_folds.loc[:, 'source'] = marking[['image_id', 'source']].groupby('image_id').min()['source']
# df_folds.loc[:, 'stratify_group'] = np.char.add(
#     df_folds['source'].values.astype(str),
#     df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str)
# )
# df_folds.loc[:, 'fold'] = 0

# for fold_number, (train_index, val_index) in enumerate(skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
#     df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number


def get_valid_transforms():
    return A.Compose([
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)


# device = torch.device('cuda:0')

# def make_ensemble_predictions(images):
#     images = list(image.to(device) for image in images)    
#     result = []
#     for net in models:
#         outputs = net(images)
#         result.append(outputs)
#     return result

# def run_wbf(predictions, image_index, image_size=512, iou_thr=0.55, skip_box_thr=0.7, weights=None):
#     boxes = [prediction[image_index]['boxes'].data.cpu().numpy()/(image_size-1) for prediction in predictions]
#     scores = [prediction[image_index]['scores'].data.cpu().numpy() for prediction in predictions]
#     labels = [np.ones(prediction[image_index]['scores'].shape[0]) for prediction in predictions]
#     boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
#     boxes = boxes*(image_size-1)
#     return boxes, scores, labels




# for j, (images, image_ids) in enumerate(data_loader):
#     if j > 0:
#         break
# predictions = make_ensemble_predictions(images)

# i = 1
# sample = images[i].permute(1,2,0).cpu().numpy()
# boxes, scores, labels = run_wbf(predictions, image_index=i)
# boxes = boxes.astype(np.int32).clip(min=0, max=511)

# fig, ax = plt.subplots(1, 1, figsize=(16, 8))

# for box in boxes:
#     cv2.rectangle(sample,
#                   (box[0], box[1]),
#                   (box[2], box[3]),
#                   (220, 0, 0), 2)
    
# ax.set_axis_off()
# ax.imshow(sample)





# import numpy as np

# first = pd.read_csv('text_results_1.npy')
# second =pd.read_csv('text_results.npy')

# np.save('two_results', np.concatenate((first, second), axis=1))



#!/usr/bin/python3  

import pandas as pd
import glob
import os


all_files = glob.glob(os.path.join('two',"*.npy"))    # 遍历当前目录下的所有以.csv结尾的文件
all_data_frame = []
row_count = 0
for file in all_files:
    data_frame = pd.read_csv(file)
    all_data_frame.append(data_frame)
    # axis=0纵向合并 axis=1横向合并
data_frame_concat = pd.concat(all_data_frame, axis=0, ignore_index=True, sort=True)
data_frame_concat.to_csv("./wine.csv", index=False, encoding="utf-8")     # 将重组后的数据集重新写入一个文件

