from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.backend import clear_session
import os
import numpy as np
import time
from tqdm import tqdm
import cv2
import TrainStep
import lovasz_losses_tf
import models
from tensorflow import keras
import matplotlib.pyplot as plt
import denseunet
import resunext
import pspnet
import mobilenet
from TransUnet import get_transunet,build_model

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
print(tf.__version__)

original_path = './subimage'
mask_path = './subimage_label'

model_results_dir = './model_results_dir'
if not os.path.exists(model_results_dir):
    os.makedirs(model_results_dir)

pretrained_weights = None
model_name = 'Transunet'

img_resize = (224,224)
input_size = (224,224,3)
n_class = 3

epochs = 350
batchSize = 16
learning_rate = 0.0005

learning_rate_base = learning_rate
total_steps = epochs
warmup_learning_rate = 0
warmup_steps = 0
hold_base_rate_steps = 0

original_files = next(os.walk(original_path))[2]
original_files.sort()

X = []
Y = []
save_name = []

for img_fl in tqdm(original_files):
    image = cv2.imread(original_path + '\\' + img_fl, cv2.IMREAD_COLOR)
    resized_img = cv2.resize(image, img_resize)
    X.append(resized_img)
    img_name_img = img_fl.split('.')[0]
    save_name.append(img_name_img)

# 单通道直接读取
    img_mask = cv2.imread(mask_path + '\\' + img_fl, cv2.IMREAD_GRAYSCALE)
    label = cv2.resize(img_mask, img_resize)
    label = label / 128
    label[label > 1.5] = 2
    label = np.round(label)

    Y.append(label)

X = np.array(X)
Y = np.array(Y)
save_name = np.array(save_name)

np.random.seed(1500)
shuffle_indices = np.random.permutation(np.arange(len(Y)))
x_shuffled = X[shuffle_indices]
y_shuffled = Y[shuffle_indices]
save_name_shuffled = save_name[shuffle_indices]

x_shuffled = X
y_shuffled = Y
save_name_shuffled = save_name

x_shuffled = x_shuffled / 255
y_shuffled = y_shuffled / 128
y_shuffled = np.round(y_shuffled,0)

print(x_shuffled.shape)
print(y_shuffled.shape)

length = int(float(len(x_shuffled))/5)


for i in range(0,5):
    tic = time.ctime()
    fp = open(model_results_dir +'\\jaccard-{}.txt'.format(i),'w')
    fp.write(str(tic) + '\n')
    fp.close()
    fp = open(model_results_dir +'\\dice-{}.txt'.format(i),'w')
    fp.write(str(tic) + '\n')
    fp.close()
    fp = open(model_results_dir +'\\class1-jaccard-{}.txt'.format(i),'w')
    fp.write(str(tic) + '\n')
    fp.close()
    fp = open(model_results_dir +'\\class2-jaccard-{}.txt'.format(i),'w')
    fp.write(str(tic) + '\n')
    fp.close()

    fp = open(model_results_dir +'\\best-jaccard-{}.txt'.format(i),'w')
    fp.write('-1.0')
    fp.close()
    fp = open(model_results_dir +'\\best-dice-{}.txt'.format(i),'w')
    fp.write('-1.0')
    fp.close()
    fp = open(model_results_dir +'\\best-class1-jaccard-{}.txt'.format(i),'w')
    fp.write('-1.0')
    fp.close()
    fp = open(model_results_dir +'\\best-class2-jaccard-{}.txt'.format(i),'w')
    fp.write('-1.0')
    fp.close()

    index = int(float(len(x_shuffled))*(i+1)/5)
    x_train = np.concatenate((x_shuffled[:index-length], x_shuffled[index:]), axis=0)
    x_val = x_shuffled[index-length:index]
    y_train = np.concatenate((y_shuffled[:index-length],y_shuffled[index:]), axis=0)
    y_val = y_shuffled[index-length:index]
    save_name_img = save_name_shuffled[index-length:index]

    # model = models.unet_bn(pretrained_weights = None ,input_size = input_size)
    config = get_transunet()
    model = build_model(config)
    model.build(input_shape=(None, 224,224,3))
    model.summary()
    print ('iter: %s' % (str(i)))
    model.compile(optimizer='adam', loss=lovasz_losses_tf.lovasz_softmax, metrics=['accuracy'])
    TrainStep.trainStep(model, x_train, y_train, x_val, y_val, epochs=epochs, batchSize=batchSize, iters = i, results_save_path = model_results_dir, learning_rate_base=learning_rate_base, warmup_learning_rate=warmup_learning_rate, warmup_steps=warmup_steps, hold_base_rate_steps=hold_base_rate_steps, n_class=n_class)


    fp = open(model_results_dir +'\\best-jaccard-{}.txt'.format(i),'r')
    best = fp.read()
    print('Best Average IoU : ', best)
    fp.close()
    fp = open(model_results_dir +'\\epoch_best-jaccard.txt','a')
    tic = time.ctime()
    fp.write('iter: ' + str(i) + '\n' + str(tic) + ':   ' + str(best) + '\n')
    fp.close()

    fp = open(model_results_dir +'\\best-dice-{}.txt'.format(i),'r')
    best = fp.read()
    print('Best Average Dice : ', best)
    fp.close()
    fp = open(model_results_dir +'\\epoch_best-dice.txt','a')
    fp.write('iter: ' + str(i) + '\n' + str(tic) + ':   ' + str(best) + '\n')
    fp.close()
