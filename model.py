import numpy as np

import keras
import keras.backend as K
import tensorflow as tf
# from DRN_Keras.drn import build_DRN26, build_DRN42
from keras.applications.resnet50 import ResNet50
from keras import regularizers
from keras.layers import (Activation, BatchNormalization, Conv2D, Dropout,
                          Input, Lambda, Layer, MaxPooling2D, ZeroPadding2D,
                          add, concatenate, multiply)
from keras.models import Model

RESIZE_FACTOR = 2

def resize_bilinear(x):
    return tf.image.resize_bilinear(x, size=[K.shape(x)[1]*RESIZE_FACTOR, K.shape(x)[2]*RESIZE_FACTOR])

def resize_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 4
    shape[1] *= RESIZE_FACTOR
    shape[2] *= RESIZE_FACTOR
    return tuple(shape)

class EAST_model:

    def __init__(self, input_size=512):
        input_image = Input(shape=(None, None, 3), name='input_image')
        overly_small_text_region_training_mask = Input(shape=(None, None, 1), name='overly_small_text_region_training_mask')
        text_region_boundary_training_mask = Input(shape=(None, None, 1), name='text_region_boundary_training_mask')
        target_score_map = Input(shape=(None, None, 1), name='target_score_map')
        resnet = ResNet50(input_tensor=input_image, weights='imagenet', include_top=False, pooling=None)
        x = resnet.get_layer('activation_49').output

        x = Lambda(resize_bilinear, name='resize_1')(x)
        x = concatenate([x, resnet.get_layer('activation_40').output], axis=3)
        x = Conv2D(128, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)

        x = Lambda(resize_bilinear, name='resize_2')(x)
        x = concatenate([x, resnet.get_layer('activation_22').output], axis=3)
        x = Conv2D(64, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)

        x = Lambda(resize_bilinear, name='resize_3')(x)
        x = concatenate([x, ZeroPadding2D(((1, 0),(1, 0)))(resnet.get_layer('activation_10').output)], axis=3)
        x = Conv2D(32, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)
        x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)

        x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)

        pred_score_map = Conv2D(1, (1, 1), activation=tf.nn.sigmoid, name='pred_score_map')(x)
        rbox_geo_map = Conv2D(4, (1, 1), activation=tf.nn.sigmoid, name='rbox_geo_map')(x) 
        rbox_geo_map = Lambda(lambda x: x * input_size)(rbox_geo_map)
        angle_map = Conv2D(1, (1, 1), activation=tf.nn.sigmoid, name='rbox_angle_map')(x)
        angle_map = Lambda(lambda x: (x - 0.5) * np.pi / 2)(angle_map)
        pred_geo_map = concatenate([rbox_geo_map, angle_map], axis=3, name='pred_geo_map')

        model = Model(inputs=[input_image, overly_small_text_region_training_mask, text_region_boundary_training_mask, target_score_map], outputs=[pred_score_map, pred_geo_map])

        self.model = model
        self.input_image = input_image
        self.overly_small_text_region_training_mask = overly_small_text_region_training_mask
        self.text_region_boundary_training_mask = text_region_boundary_training_mask
        self.target_score_map = target_score_map
        self.pred_score_map = pred_score_map
        self.pred_geo_map = pred_geo_map


class EAST_DRN_model():
    def __init__(self, input_size=512):
        input_image = Input(shape=(None, None, 3), name='input_image')
        overly_small_text_region_training_mask = Input(shape=(None, None, 1), name='overly_small_text_region_training_mask')
        text_region_boundary_training_mask = Input(shape=(None, None, 1), name='text_region_boundary_training_mask')
        target_score_map = Input(shape=(None, None, 1), name='target_score_map')
        # drn_backbone = build_DRN26(input_tensor=input_image)
        drn_backbone = build_DRN42(input_tensor=input_image)
        x = drn_backbone.output

        x = Lambda(resize_bilinear, name='resize_1')(x)
        # x = concatenate([x, resnet.get_layer('activation_40').output], axis=3)
        x = Conv2D(128, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)

        x = Lambda(resize_bilinear, name='resize_2')(x)
        # x = concatenate([x, resnet.get_layer('activation_22').output], axis=3)
        x = Conv2D(64, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)

        x = Lambda(resize_bilinear, name='resize_3')(x)
        # x = concatenate([x, ZeroPadding2D(((1, 0),(1, 0)))(resnet.get_layer('activation_10').output)], axis=3)
        x = Conv2D(32, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)
        x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)

        x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)

        pred_score_map = Conv2D(1, (1, 1), activation=tf.nn.sigmoid, name='pred_score_map')(x)
        rbox_geo_map = Conv2D(4, (1, 1), activation=tf.nn.sigmoid, name='rbox_geo_map')(x) 
        rbox_geo_map = Lambda(lambda x: x * input_size)(rbox_geo_map)
        angle_map = Conv2D(1, (1, 1), activation=tf.nn.sigmoid, name='rbox_angle_map')(x)
        angle_map = Lambda(lambda x: (x - 0.5) * np.pi / 2)(angle_map)
        pred_geo_map = concatenate([rbox_geo_map, angle_map], axis=3, name='pred_geo_map')

        model = Model(inputs=[input_image, overly_small_text_region_training_mask, text_region_boundary_training_mask, target_score_map], outputs=[pred_score_map, pred_geo_map])

        self.model = model
        self.input_image = input_image
        self.overly_small_text_region_training_mask = overly_small_text_region_training_mask
        self.text_region_boundary_training_mask = text_region_boundary_training_mask
        self.target_score_map = target_score_map
        self.pred_score_map = pred_score_map
        self.pred_geo_map = pred_geo_map

def basic_block(input_tensor, output_channels, name, strides=1, dilation=1, kernel=3, residual=True, repeat=1):
    """
    Defines the basic block as used in the DRN. 

    Inputs:
        - input_tensor: Inputted tensor
        - output_channels: Channel size of output tensor
        - name: Name for the block
        - strides: Stride length of convolutional filter
        - dilation: Dilation rate of convolutional filter
        - kernel: Kernel size of convolutional filter
        - residual: Flag indicating whether the residual connection is used or not
        - repeat: Number of times to repeat block
    
    Outputs:
        - model: BasicBlock Keras tensor
    """

    # input_tensor = Input(shape=(None, None, input_channels))
    ## save input tensor for residual connection
    shortcut = input_tensor
    x = input_tensor

    ## first conv level
    if residual:
        y = SamePadConv2D(x, output_channels, name + "_conv1", strides=strides, dilation=dilation, kernel=kernel)
    else:
        y = Conv2D(output_channels, (kernel, kernel), strides=(strides, strides), dilation_rate=(dilation, dilation), padding="same", name=name + "_conv1")(x)
    
    y = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True, name=name + "bn_1")(y)
    y = Activation('relu', name=name + "relu_1")(y)

    for i in range(repeat):
        ## second conv level
        if residual:
            y = SamePadConv2D(y, output_channels, name + "_conv" + str(i), strides=strides, dilation=dilation, kernel=kernel)
        else:
            y = Conv2D(output_channels, (kernel, kernel), strides=(strides, strides), dilation_rate=(dilation, dilation), padding="same", name=name + "_conv"  + str(i))(y)

        y = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True, name=name + "bn_2" + str(i))(y)

    ## optional residual connection
    if residual:
        y = Add(name=name + "add")([shortcut, y])
    
    y = Activation('relu', name=name + "relu_2")(y)

    ## return block
    return y


def SamePadConv2D(input_tensor, output_channels, name, strides=1, dilation=1, kernel=3):
    """
    Returns output tensor with same size
    Inputs:
        - input_tensor: Inputted tensor
        - output_channels: Channel size of output tensor
        - name: Name for the block
        - strides: Stride length of convolutional filter
        - dilation: Dilation rate of convolutional filter
        - kernel: Kernel size of convolutional filter
        - residual: Flag indicating whether the residual connection is used or not
        - repeat: Number of times to repeat block
    
    Outputs:
        - y: Padded and convoluted output tensor
    """
    ## get input size of tensor
    input_size = K.int_shape(input_tensor)[1]
    # Ensures padding = 'SAME' for ODD kernel selection 
    padding = ((strides * (input_size - 1)) - input_size + (dilation * (kernel - 1))) // 2

    padded = ZeroPadding2D(padding=padding)(input_tensor)
    output = Conv2D(output_channels, (kernel, kernel), 
        padding="VALID", strides=1, dilation_rate=dilation, name=name)(padded)
    
    return output
    



def build_DRN42(input_tensor=None):
    """
    Creates the DRNC_42 Model.
    Inputs:
        None
    Outputs:
        - model: DRNC_42 Keras Model
    """
    if input_tensor is None:
        input_tensor = Input(shape=(None, None, 3))

    ## input block
    y = Conv2D(16, (7,7))(input_tensor)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    ## layer 1
    y = basic_block(y, 16, "layer1", residual=False)

    ## layer 2
    y = basic_block(y, 32, "layer2", strides=2, residual=False)

    ## layer 3
    y = basic_block(y, 64, "layer3_1", strides=2, residual=False)
    y = basic_block(y, 64, "layer3_2", strides=2, repeat=3)

    ## layer 4
    y = basic_block(y, 128, "layer4_1", strides=2, residual=False)
    y = basic_block(y, 128, "layer4_2", strides=2, repeat=4)

    ## layer 5
    y = basic_block(y, 256, "layer5_1", dilation=2, residual=False)
    y = basic_block(y, 256, "layer5_2", dilation=2, repeat=6)

    ## layer 6
    y = basic_block(y, 512, "layer6_1", dilation=4, residual=False)
    y = basic_block(y, 512, "layer6_2", dilation=4, repeat=3)

    ## layer 7
    y = basic_block(y, 512, "layer7", dilation=2, residual=False)

    ## layer 8
    y = basic_block(y, 512, "layer8", residual=False)

    return keras.models.Model(inputs=input_tensor, outputs=y)