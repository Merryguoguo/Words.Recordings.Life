from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import glob
from random import shuffle
import cv2
from PIL import Image
import matplotlib.image as mpimg
# import gdal
# from data_argumentation import *
from tensorflow.keras.utils import to_categorical
from pylab import *
from imgaug import augmenters as iaa
# import imgaug as ia
import numpy as np
from scipy import misc
import os
import pdb

from scipy.ndimage.measurements import label


# path: '/mnt/data/guoyujun/RICE2'
def generatedata_0riginal_train(path,batchsize):
    imgs = glob.glob(path + "label/*.png")
    train  = len(imgs)*0.8
    val = len(imgs)*0.1

    images = [] # list for Multi-channel Cloudy
    cleans = [] # list for Multi-channel Clean
    masks = [] # list for Multi-channel Mask

    cnt = 0

    while 1:
        for imgname in imgs[0:int(train)]: 
            
            midname = imgname[imgname.rindex("/") + 1:]
            name = midname.split('.')[0]
            
            # # 真实云污染数据
            # img = mpimg.imread(path + "cloud/" + name + '.png') 
            # # 模拟云污染数据
            img = mpimg.imread(path + "simulated_stscnn_mul/" + name + '.png') 
            cln = mpimg.imread(path + "label/" + name + '.png') 
            msk = mpimg.imread(path + "stscnn_mask_mul_0/" + name + '.png') 

            # 数据处理应发生在这个部分
            # cloudy image 
            img = img_to_array(img).astype('float32')
            img /= 255
            # clean image
            cln = img_to_array(cln).astype('float32')
            cln /= 255
            # mask
            msk = img_to_array(msk).astype('float32')
            msk /= 255
            msk = np.concatenate([msk, msk, msk], 2)

            # # cloudy image 
            # img = cln * msk

            # cloud = image - clean
            images.append(img)
            cleans.append(cln)
            masks.append(msk)
            # clouds.append(cloud)

            cnt += 1
            if cnt == batchsize:
                imagedatas = np.asarray(images)
                labeldatas = np.asarray(cleans)
                maskdatas = np.asarray(masks)

                yield ([imagedatas, maskdatas],labeldatas) # 运行成功

                cnt = 0

                images = []
                cleans = []
                masks = []


def generatedata_0riginal_val(path,batchsize):
    imgs = glob.glob(path + "label/*.png")
    train  = len(imgs)*0.8
    val = len(imgs)*0.1

    images = [] # list for Multi-channel Cloudy
    cleans = [] # list for Multi-channel Clean
    masks = [] # list for Multi-channel Mask

    cnt = 0

    while 1:
        for imgname in imgs[int(train):(int(train) + int(val))]: 
            # pdb.set_trace()
            midname = imgname[imgname.rindex("/") + 1:]
            name = midname.split('.')[0]
            
            # # 真实云污染数据
            # img = mpimg.imread(path + "cloud/" + name + '.png') 
            # # 模拟云污染数据
            img = mpimg.imread(path + "simulated_stscnn_mul/" + name + '.png') 
            cln = mpimg.imread(path + "label/" + name + '.png') 
            msk = mpimg.imread(path + "stscnn_mask_mul_0/" + name + '.png') 

            # 数据处理应发生在这个部分
            # cloudy image 
            img = img_to_array(img).astype('float32')
            img /= 255
            # clean image
            cln = img_to_array(cln).astype('float32')
            cln /= 255
            # mask
            msk = img_to_array(msk).astype('float32')
            msk /= 255
            msk = np.concatenate([msk, msk, msk], 2)

            # # cloudy image 
            # img = cln * msk

            images.append(img)
            cleans.append(cln)
            masks.append(msk)

            cnt += 1
            if cnt == batchsize:
                imagedatas = np.asarray(images)
                labeldatas = np.asarray(cleans)
                maskdatas = np.asarray(masks)

                yield ([imagedatas, maskdatas],labeldatas) # 运行成功

                cnt = 0

                images = []
                cleans = []
                masks = []


def generatedata_0riginal_test(path,batchsize):
    imgs = glob.glob(path + "label/*.png")
    train  = len(imgs)*0.8
    val = len(imgs)*0.1

    images = [] # list for Multi-channel Cloudy
    cleans = [] # list for Multi-channel Clean
    masks = [] # list for Multi-channel Mask

    cnt = 0

    while 1:
        for imgname in imgs[(int(train) + int(val)):]: 
            # pdb.set_trace()
            midname = imgname[imgname.rindex("/") + 1:]
            name = midname.split('.')[0]
            
            # # 真实云污染数据
            # img = mpimg.imread(path + "cloud/" + name + '.png') 
            # # 模拟云污染数据
            img = mpimg.imread(path + "simulated_stscnn_mul/" + name + '.png') 
            cln = mpimg.imread(path + "label/" + name + '.png') 
            msk = mpimg.imread(path + "stscnn_mask_mul_0/" + name + '.png') 

            # 数据处理应发生在这个部分
            # cloudy image 
            img = img_to_array(img).astype('float32')
            img /= 255
            # clean image
            cln = img_to_array(cln).astype('float32')
            cln /= 255
            # mask
            msk = img_to_array(msk).astype('float32')
            msk /= 255
            msk = np.concatenate([msk, msk, msk], 2)

            # # cloudy image 
            # img = cln * msk

            images.append(img)
            cleans.append(cln)
            masks.append(msk)

            cnt += 1
            if cnt == batchsize:
                imagedatas = np.asarray(images)
                labeldatas = np.asarray(cleans)
                maskdatas = np.asarray(masks)

                yield ([imagedatas, maskdatas],labeldatas) # 运行成功

                cnt = 0

                images = []
                cleans = []
                masks = []

