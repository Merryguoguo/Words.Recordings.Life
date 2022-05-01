import gc
from copy import deepcopy
import tensorflow.keras
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image
# 模型导入
from MSUNET import CloudRemoval
from Indicators import mPSNR, mSSIM, SAM, CC
import cv2
import glob
import data
import math
import matplotlib.image as mpimg
import os
import tensorflow as tf
import pdb
# from LS_Loss import LS_loss, loss_l1, loss_L1, R_loss
from tensorflow.keras import backend as K


## 该代码处理的是批量模拟云污染情况 

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['CUDA_VISIBLE_DEVICES'] = '3' 
print('-------------------------------GPU---------------------------')	
tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
print('available?', tf.test.is_gpu_available())
print('CUDA?', tf.test.is_built_with_cuda())
# pdb.set_trace()
print('--------------------------------------------------------------')


MAX_BATCH_SIZE = 128

patch_size = 512

# 模拟云污染
Test_DIR = "/mnt/data/guoyujun/RICE2"

# -------------------------------------测试结果保存路径-------------------------------------
# UNET 100
Result_DIR_refine = "/mnt/data/guoyujun/Unet/results" 

# 单输出
if not os.path.exists(Result_DIR_refine):
    os.makedirs(Result_DIR_refine)


# ---------------------------------------------模型加载--------------------------------------------
model = CloudRemoval()
model.summary()
# pdb.set_trace()
# 模型引入路径

# # UNET 500 RICE2_STSCNNMask_replace
# model.load('/mnt/data/guoyujun/MSFCN/CR/RICE2_DATA/UNet/bestweight/472_accuracy_0.89.h5')

# UNET 500 RICE2_STSCNNMask_Multiply
model.load('/mnt/data/guoyujun/MSFCN/CR/RICE2_DATA/UNet/bestweight/442_accuracy_0.89.h5')


# ---------------------------------------------数据以及预测--------------------------------------------
# 模拟
imgsname = glob.glob(Test_DIR + "/simulated_stscnn_mul/*.png")
train  = len(imgsname)*0.8
val = len(imgsname)*0.1

imgdatas = np.ndarray((1, patch_size, patch_size, 3), dtype=np.float32)
maskdatas = np.ndarray((1, patch_size, patch_size, 3), dtype=np.float32)   
originaldatas = np.ndarray((1, patch_size, patch_size, 3), dtype=np.float32)   

psnrs = []
ssims = []
sams = []
ccs = []
names = []
num = 0

for imgname in imgsname[0:5]:
# for imgname in imgsname[(int(train) + int(val)):(int(train) + int(val) + 5)]:
    name = imgname[imgname.rindex("/") + 1:]
    names.append(name)

    # 模拟云污染代码
    img = mpimg.imread(Test_DIR + "/simulated_stscnn_mul/" + name)
    mask = mpimg.imread(Test_DIR + "/stscnn_mask_mul_0/" + name)
    label = mpimg.imread(Test_DIR + "/label/" + name)

    img = img_to_array(img).astype('float32')
    img /= 255
    mask = img_to_array(mask).astype('float32')
    mask /= 255
    label = img_to_array(label).astype('float32')
    label /= 255

    mask = np.concatenate([mask, mask, mask], 2)

    imgdatas[0] = img 
    # maskdatas[0] = 1 - (mask)     
    maskdatas[0] = mask
    originaldatas[0] = label

    # 利用模型测试
    # 双输入
    # model.summary()
    # pdb.set_trace()
    prediction = model.predict([imgdatas, originaldatas])

    # 精度指标 真实云污染时没有这一部分
    # 单输出
    psnr = mPSNR(originaldatas[0,:,:,:], prediction[0,:,:,:])
    ssim = mSSIM(originaldatas[0,:,:,:], prediction[0,:,:,:])
    sam = SAM(originaldatas[0,:,:,:], prediction[0,:,:,:])
    cc = CC(originaldatas[0,:,:,:], prediction[0,:,:,:])
    psnrs.append(psnr)
    ssims.append(ssim)
    sams.append(sam)
    ccs.append(cc)

    # 保存预测结果
    # 单输出
    # pdb.set_trace()
    prediction = prediction[0, :, :, :]
    prediction_img = array_to_img(prediction)
    prediction_img.save(Result_DIR_refine + "/%s" % name)

    img = imgdatas[0, :, :, :]
    img = array_to_img(img)
    img.save(Result_DIR_refine + "/img_%s" % name)

    mask = maskdatas[0, :, :, :]
    mask_img = array_to_img(mask)
    mask_img.save(Result_DIR_refine + "/mask_%s" % name)

    label = originaldatas[0, :, :, :]
    label_img = array_to_img(label)
    label_img.save(Result_DIR_refine + "/label_%s" % name)

    num = num + 1
    print(num)


# 单输出
fileObject = open(Result_DIR_refine + '/psnr.txt', 'w')
for ip in psnrs:
    ip = str(ip)
    fileObject.write(ip)
    fileObject.write('\n')
fileObject.close()

fileObject = open(Result_DIR_refine + '/ssim.txt', 'w')
for ip in ssims:
    ip = str(ip)
    fileObject.write(ip)
    fileObject.write('\n')
fileObject.close()

fileObject = open(Result_DIR_refine + '/sam.txt', 'w')
for ip in sams:
    ip = str(ip)
    fileObject.write(ip)
    fileObject.write('\n')
fileObject.close()

fileObject = open(Result_DIR_refine + '/cc.txt', 'w')
for ip in ccs:
    ip = str(ip)
    fileObject.write(ip)
    fileObject.write('\n')
fileObject.close()

fileObject = open(Result_DIR_refine + '/name.txt', 'w')
for ip in names:
    ip = str(ip)
    fileObject.write(ip)
    fileObject.write('\n')
fileObject.close()

