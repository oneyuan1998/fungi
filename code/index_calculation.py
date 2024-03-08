import cv2
import numpy as np
import tensorflow as tf
import os
from scipy.spatial.distance import directed_hausdorff,cdist
import scipy
from scipy.ndimage.morphology import distance_transform_edt
from tqdm import *
import math


def dice_iou(y_true, y_pred):
    y_true_f = y_true.ravel()
    y_pred_f = y_pred.ravel()
    intersection = np.sum(y_true_f * y_pred_f)
    sum = np.sum(y_true_f) + np.sum(y_pred_f)
    smooth = 0.00001
    dice = (2. * intersection ) / (sum + smooth)
    iou = dice / (2 - dice)
    return dice, iou

def perf_measure(y_actual, y_hat):
    y_actual = y_actual.ravel()
    y_hat = y_hat.ravel()

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

def cal_hd_assd(seg_image1, seg_image2):
    # Convert segmented images to binary masks
    segmentation_boundary = scipy.ndimage.morphology.binary_dilation(seg_image1 ) - seg_image1 
    ground_truth_boundary = scipy.ndimage.morphology.binary_dilation(seg_image2 ) - seg_image2 

    # Get the coordinates of the segmented regions in each image
    coords1 = np.argwhere(segmentation_boundary)
    coords2 = np.argwhere(ground_truth_boundary)

    if len(coords1) == 0 or len(coords2) == 0:
        return 0,0

    # Calculate the pairwise Euclidean distances between the coordinates
    distances1 = cdist(coords1, coords2, 'euclidean')
    distances2 = cdist(coords2, coords1, 'euclidean')

    # Calculate the Hausdorff distance as the maximum distance from image1 to image2
    hausdorff_distance1 = np.max(np.min(distances1, axis=1))
    hausdorff_distance2 = np.max(np.min(distances2, axis=1))

    hausdorff_distance = max(hausdorff_distance1, hausdorff_distance2)

    assd1 = np.min(distances1, axis=1)
    assd2 = np.min(distances2, axis=1)

    assd = (np.sum(assd1) + np.sum(assd2)) / (np.sum(segmentation_boundary ) + np.sum(ground_truth_boundary))

    return hausdorff_distance, assd

predict_dir = './predict_dir'
mask_dir = './subimage_label'
img_resize = (224,224)

predict_files = next(os.walk(predict_dir))[2]
predict_files.sort()

mask_files = next(os.walk(mask_dir))[2]
mask_files.sort()

X = []
Y = []

for img_fl in tqdm(predict_files):
    image = cv2.imread(predict_dir + '\\' + img_fl, cv2.IMREAD_COLOR)
    resized_img = cv2.resize(image, img_resize)
    X.append(resized_img)

    image2 = cv2.imread(mask_dir + '\\' + img_fl, cv2.IMREAD_GRAYSCALE)
    resized_img2 = cv2.resize(image2, img_resize)
    resized_img2 = resized_img2 / 128
    resized_img2[resized_img2 > 1.5] = 2
    resized_img2 = np.round(resized_img2)
    Y.append(resized_img2)

X = np.array(X)
X = X / 255
X = np.round(X)
Y = np.array(Y)
Y = np.round(Y)

dice = 0
jaccard = 0
Precision = 0
classIou = []
hd = []
assd = []
n_class = 3
for index in range(1, n_class):
    yt = Y.copy()
    yt[yt != index] = 0
    yt = yt / index
    yt = np.expand_dims(yt, axis=-1)

    dice_ret, jaccard_ret = dice_iou(yt, X[:,:,:,index])
    classIou.append(jaccard_ret)
    dice += dice_ret
    jaccard += jaccard_ret
    print('jaccard{}:'.format(index)+str(jaccard_ret))
    print('dice{}:'.format(index)+str(dice_ret))
    
    TP, FP, TN, FN = perf_measure(yt, X[:,:,:,index])
    Precision_ret = TP/(TP+FP)
    Precision += Precision_ret
    print('Precision{}:'.format(index)+str(Precision_ret))
    Recall = TP/(TP+FN)
    print('Recall{}:'.format(index)+str(Recall))
    for j in trange(0, yt.shape[0]):
        HD_cal, ASSD_cal = cal_hd_assd(yt[j,:,:], X[j,:,:,index])
        if HD_cal != 0:
            hd.append(HD_cal)

        if ASSD_cal != 0:
            assd.append(ASSD_cal)

    hdAve = np.mean(hd)
    assdAve = np.mean(assd)
    print('HD{}:'.format(index)+str(hdAve))
    print('ASSD{}:'.format(index)+str(assdAve))
    hd = []
    assd = []

    Matthew_correlation_coefficient = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    print('matthews_corrcoef{}:'.format(index)+str(Matthew_correlation_coefficient))




