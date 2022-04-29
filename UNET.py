from termios import B50
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, AveragePooling2D, Conv2DTranspose, Add, \
    Cropping2D, ZeroPadding2D, Activation, Concatenate, BatchNormalization, LeakyReLU, Lambda, AveragePooling2D, Multiply
from tensorflow.keras.optimizers import *
from tensorflow.keras.activations import elu, sigmoid
import tensorflow.keras as k
import tensorflow as tf
import os
from datetime import datetime
# from perpectral_loss import total_loss
# from LS_Loss import LS_loss
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Dropout, LeakyReLU, BatchNormalization, Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.applications import VGG16
from tensorflow.keras import backend as K
from subpixel_conv2d import SubpixelConv2D
from layers import ConvOffset2D
from non_local import non_local_block
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization,\
    Activation, Dropout, Softmax, Permute, Lambda
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Reshape
from sklearn import preprocessing



def LRM_L(input):
    # b, h, w, c = K.int_shape(input)

    # STGAN_DATA 
    h = 256
    w = 256
    c = 256 # h,w,c根据上一层的特征大小调整
    x1 = Conv2D(c, (1, 1), padding='same')(input)  # b * h * w  *c
    # print(x1.get_shape())

    # 这里一共下采样了  4*4*4 = 64 倍   针对 512 feature 就是 秩上限为8

    x2 = Conv2D(c // 4, (3, 3), activation='relu', padding='same')(input)
    x2 = Conv2D(c // 16, (3, 3), activation='relu', padding='same')(x2)
    x2 = Conv2D(c // 64, (1, 1))(x2)  # b * h * w  * c//64

    x3 = Conv2D(c // 4, (3, 3), activation='relu', padding='same')(input)
    x3 = Conv2D(c // 16, (3, 3), activation='relu', padding='same')(x3)
    x3 = Conv2D(c // 64, (1, 1))(x3)  # b * h * w  * c//64

    x3 = Reshape((h * w, c // 64))(x3)
    x3 = Softmax(axis=-1)(x3)  # b * hw * c//64

    x1 = Reshape((h * w, c))(x1)
    x1 = Permute((2, 1))(x1)  # b * c * hw

    x2 = Reshape((h * w, c // 64))(x2)  # b * hw * c//64

    lambda_batchdot = Lambda(lambda x: K.batch_dot(x[0], x[1]))
    x1 = lambda_batchdot([x1, x2])

    # l2 normalize
    lambda_l2nor = Lambda(lambda x: K.l2_normalize(x, axis=1))
    x1 = lambda_l2nor(x1)
    # x1 = _l2norm(x1, dim=1)  # b * c * c//64
    x1 = Permute((2, 1))(x1)  # b * c//64 * c

    x3 = lambda_batchdot([x3, x1])  # b * hw * c
    x3 = Reshape((h, w, c))(x3)  # b * hw * c

    x3 = Conv2D(c, (1, 1), activation='relu')(x3) # 相当于低秩？

    return x3

def LRM_S(input):
    # b, h, w, c = K.int_shape(input)

    # STGAN_DATA 
    h = 256
    w = 256
    c = 256 # h,w,c根据上一层的特征大小调整
    x1 = Conv2D(c, (1, 1), padding='same')(input)  # b * h * w  *c
    # print(x1.get_shape())

    # 这里一共下采样了  4*4*4 = 64 倍   针对 512 feature 就是 秩上限为8

    x2 = Conv2D(c // 4, (3, 3), activation='relu', padding='same')(input)
    x2 = Conv2D(c // 16, (3, 3), activation='relu', padding='same')(x2)
    x2 = Conv2D(c // 64, (1, 1))(x2)  # b * h * w  * c//64

    x3 = Conv2D(c // 4, (3, 3), activation='relu', padding='same')(input)
    x3 = Conv2D(c // 16, (3, 3), activation='relu', padding='same')(x3)
    x3 = Conv2D(c // 64, (1, 1))(x3)  # b * h * w  * c//64

    x3 = Reshape((h * w, c // 64))(x3)
    x3 = Softmax(axis=-1)(x3)  # b * hw * c//64

    x1 = Reshape((h * w, c))(x1)
    x1 = Permute((2, 1))(x1)  # b * c * hw

    x2 = Reshape((h * w, c // 64))(x2)  # b * hw * c//64

    lambda_batchdot = Lambda(lambda x: K.batch_dot(x[0], x[1]))
    x1 = lambda_batchdot([x1, x2])

    # l2 normalize
    lambda_l2nor = Lambda(lambda x: K.l2_normalize(x, axis=1))
    x1 = lambda_l2nor(x1)
    # x1 = _l2norm(x1, dim=1)  # b * c * c//64
    x1 = Permute((2, 1))(x1)  # b * c//64 * c

    x3 = lambda_batchdot([x3, x1])  # b * hw * c
    x3 = Reshape((h, w, c))(x3)  # b * hw * c

    x3 = Conv2D(c, (1, 1), activation='relu')(x3) # 256 256 256
    x4 = Add()([input,x3]) # 参差学习？, 相当于获得S?（待考察） # 256 256 256 

    return x4


class CloudRemoval(object):
    # def __init__(self, img_rows = 32, img_cols = 32, weight_filepath=None): 
    # def __init__(self, img_rows = 400, img_cols = 400, weight_filepath=None):
    def __init__(self, img_rows = 512, img_cols = 512, weight_filepath=None): 
    # def __init__(self, img_rows = 256, img_cols = 256, weight_filepath=None): 
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.model = self.Unet()
        self.current_epoch = 0
        self.weight_filepath = weight_filepath
        # self.vgg_layers = [3, 6, 10]
        # self.vgg = self.build_vgg()

    def build_vgg(self):
        """
        Load pre-trained VGG16 from keras applications
        Extract features to be used in loss function from last conv layer, see architecture at:
        https://github.com/keras-team/keras/blob/master/keras/applications/vgg16.py
        """
        # Input image to extract features from
        img = Input(shape=(self.img_rows, self.img_cols, 3)) #  whether to include the 3 fully-connected layers at the top of the network.

        # Get the vgg network from Keras applications
        vgg = VGG16(weights="imagenet", include_top=False)
        # VGG16(): Return A keras.Model instance.
        # 'imagenet': pre-training on ImageNet 
        # include_top: whether to include the 3 fully-connected layers at the top of the network.

        # Output the first three pooling layers
        vgg.outputs = [vgg.layers[i].output for i in self.vgg_layers] # self.vgg_layers = [3, 6, 10]

        # Create model and compile
        model = Model(inputs=img, outputs=vgg(img))
        model.trainable = False
        model.compile(loss='mse', optimizer='adam')

        return model


    def Unet(self):

        data = Input((self.img_rows, self.img_cols, 3)) 
        mask = Input((self.img_rows, self.img_cols, 3)) 

        ##--------------------------Data Fusion-------------------------------
        conv_cloudy = Conv2D(32, 3, activation='relu', padding='same')(data)
        cloudy_mask = k.layers.multiply([data, mask])
        conv_cloudy_1 = Conv2D(32, 3, activation='relu', padding='same')(cloudy_mask)

        con = Concatenate(axis=-1)([conv_cloudy, conv_cloudy_1])

        # Head
        conv_con = Conv2D(64, 3, activation='relu', strides=(1, 1), padding='same', kernel_initializer='he_normal')(con)

        # Downsample
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv_con)
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        # conv5 = Conv2D(512, 3, activation='relu', padding='same')(pool4)
        # conv5 = Conv2D(512, 3, activation='relu', padding='same')(conv5)
        # pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

        # conv6 = Conv2D(512, 3, activation='relu', padding='same')(pool5)
        # conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)
        # pool6 = MaxPooling2D(pool_size=(2, 2))(conv6)

        # conv7 = Conv2D(512, 3, activation='relu', padding='same')(pool6)
        # conv7 = Conv2D(512, 3, activation='relu', padding='same')(conv7)
        # pool7 = MaxPooling2D(pool_size=(2, 2))(conv7)


        # Upsample
        # up8 = Conv2DTranspose(512, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(pool7)
        # merge8 = k.layers.concatenate([conv7, up8], axis=3)
        # conv8 = Conv2D(512, 3, activation='relu', padding='same')(merge8)

        # up9 = Conv2DTranspose(512, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(pool6)
        # merge9 = k.layers.concatenate([conv6, up9], axis=3)
        # conv9 = Conv2D(512, 3, activation='relu', padding='same')(merge9)

        # up10 = Conv2DTranspose(512, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(pool5)
        # merge10 = k.layers.concatenate([conv5, up10], axis=3)
        # conv10 = Conv2D(512, 3, activation='relu', padding='same')(merge10)

        up11 = Conv2DTranspose(512, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(pool4)
        merge11 = k.layers.concatenate([conv4, up11], axis=3)
        conv11 = Conv2D(512, 3, activation='relu', padding='same')(merge11)
        
        up12 = Conv2DTranspose(256, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(pool3)
        merge12 = k.layers.concatenate([conv3, up12], axis=3)
        conv12 = Conv2D(256, 3, activation='relu', padding='same')(merge12)

        up13 = Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(pool2)
        merge13 = k.layers.concatenate([conv2, up13], axis=3)
        conv13 = Conv2D(128, 3, activation='relu', padding='same')(merge13)

        up14 = Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(pool1)
        merge14 = k.layers.concatenate([conv1, up14], axis=3)
        conv14 = Conv2D(64, 3, activation='relu', padding='same')(merge14)

        output = Conv2D(3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv14)

        model = Model(inputs=[data, mask], outputs=output)

        model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['accuracy'])

        return model


    def MS_Unet_s(self):

        data = Input((self.img_rows, self.img_cols, 12)) # Multitemporal Cloudy 256 256 12
        # 256 256 3*4: 三个波段；四个时相：三个云污染，一个Clear(temporally, 不是label) 

        # 初始特征提取
        conv_1 = Conv2D(30, 3, activation='relu', strides=(1, 1), padding='same', kernel_initializer='he_normal')(data)
        conv_2 = Conv2D(30, 3, activation='relu', strides=(1, 1), padding='same', kernel_initializer='he_normal')(conv_1)

        # Downsample
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv_2)
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        # Middle
        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)

        # Upsample
        up6 = Conv2DTranspose(512, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same')(conv5)
        merge6 = k.layers.concatenate([conv4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
        conv6 = Conv2D(32, 3, activation='relu', padding='same')(conv6)
        b1 = Conv2D(1, 1, activation=None, use_bias=False, padding='same')(conv6)

        up7 = Conv2DTranspose(256, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(conv6)
        merge7 = k.layers.concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
        conv7 = Conv2D(32, 3, activation='relu', padding='same')(conv7)
        b2 = Conv2D(1, 1, activation=None, use_bias=False, padding='same')(conv7)

        up8 = Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(conv7)
        merge8 = k.layers.concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
        conv8 = Conv2D(32, 3, activation='relu', padding='same')(conv8)
        b3 = Conv2D(1, 1, activation=None, use_bias=False, padding='same')(conv8)

        up9 = Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(conv8)
        merge9 = k.layers.concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
        conv9 = Conv2D(32, 3, activation='relu', padding='same')(conv9)
        b4 = Conv2D(1, 1, activation=None, use_bias=False, padding='same', name='b4')(conv9)

        ob1 = UpSampling2D(size=(8, 8), data_format=None)(b1)
        ob2 = UpSampling2D(size=(4, 4), data_format=None)(b2)
        ob3 = UpSampling2D(size=(2, 2), data_format=None)(b3)

        fuse = Concatenate(axis=-1)([ob1, ob2, ob3, b4])

        output = Conv2D(12, 3, activation='relu', padding='same', kernel_initializer='he_normal')(fuse)

        model = Model(inputs=[data], outputs=output)

        model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['accuracy'])

        return model


    def fit(self, generator, epochs=10, plot_callback=None, *args, **kwargs):
        """Fit the U-Net to a (images, targets) generator

        param generator: training generator yielding (maskes_image, original_image) tuples
        param epochs: number of epochs to train for
        param plot_callback: callback function taking Unet model as parameter
        """

        # Loop over epochs
        for _ in range(epochs):

            # Fit the model
            self.model.fit_generator(
                generator,
                epochs=self.current_epoch + 1,
                initial_epoch=self.current_epoch,
                use_multiprocessing=True,
                *args, **kwargs 
            )

            # Update epoch
            self.current_epoch += 1

            # After each epoch predict on test images & show them
            if plot_callback:
                plot_callback(self.model)

            # Save logfile
            if self.weight_filepath:
                self.save()

    def predict(self, sample):
        """Run prediction using this model"""
        return self.model.predict(sample)

    def summary(self):
        """Get summary of the UNet model"""
        print(self.model.summary())

    def save(self):
        self.model.save_weights(self.current_weightfile())

    def load(self, filepath, train_bn=True, lr=0.0002):

        self.model = self.Unet()

        # Load weights into model
        epoch = int(os.path.basename(filepath).split("_")[0])
        assert epoch > 0, "Could not parse weight file. Should start with 'X_', with X being the epoch"
        self.current_epoch = epoch
        self.model.load_weights(filepath)

    def current_weightfile(self):
        assert self.weight_filepath != None, 'Must specify location of logs'
        return self.weight_filepath + "{}_weights_{}.h5".format(self.current_epoch, self.current_timestamp())

    @staticmethod
    def current_timestamp():
        return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    @staticmethod
    def l1(y_true, y_pred):
        """Calculate the L1 loss used in all loss calculations"""
        if K.ndim(y_true) == 4:
            return K.sum(K.abs(y_pred - y_true), axis=[1, 2, 3])
        elif K.ndim(y_true) == 3:
            return K.sum(K.abs(y_pred - y_true), axis=[1, 2])
        else:
            raise NotImplementedError("Calculating L1 loss on 1D tensors? should not occur for this network")

    @staticmethod
    def gram_matrix(x, norm_by_channels=False):
        """Calculate gram matrix used in style loss"""

        # Assertions on input
        assert K.ndim(x) == 4, 'Input tensor should be a 4d (B, H, W, C) tensor'
        assert K.image_data_format() == 'channels_last', "Please use channels-last format"

        # Permute channels and get resulting shape
        x = K.permute_dimensions(x, (0, 3, 1, 2))
        shape = K.shape(x)
        B, C, H, W = shape[0], shape[1], shape[2], shape[3]

        # Reshape x and do batch dot product
        features = K.reshape(x, K.stack([B, C, H * W]))
        gram = K.batch_dot(features, features, axes=2)

        # Normalize with channels, height and width
        gram = gram / K.cast(C * H * W, x.dtype)

        return gram



