

import fire
# import keras.backend as K
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Conv2D, Conv2DTranspose,Reshape, Dense, Flatten, Input, Lambda, MaxPooling2D,Activation,GlobalAveragePooling2D,GlobalMaxPooling2D
from tensorflow.keras.layers import UpSampling2D, add, concatenate

# WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
#                 'releases/download/v0.1/'
#                 'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
# WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
#                        'releases/download/v0.1/'
#                        'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')


import os, sys
import keras_applications
from tensorflow.keras import applications
def siam_unet_vgg16(inputshape=(256, 256, 3)):
    input_t1 = Input(inputshape, name='Input_t1')
    input_t2 = Input(inputshape, name='Input_t2')

    vgg_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=inputshape)

    # w1 = model_left.get_weights()
    # print(vgg_model.summary())

    b5c3_model = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('block5_conv3').output)
    b5c3_model.trainable = False

    b4c3_model = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('block4_conv3').output)
    b4c3_model.trainable = False

    b3c3_model = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('block3_conv3').output)
    b3c3_model.trainable = False

    b2c2_model = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('block2_conv2').output)
    b2c2_model.trainable = False

    b1c2_model = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('block1_conv2').output)
    b1c2_model.trainable = False

    t1_b5c3 = b5c3_model(input_t1)
    t2_b5c3 = b5c3_model(input_t2)

    t1_b4c3 = b4c3_model(input_t1)
    t2_b4c3 = b4c3_model(input_t2)

    t1_b3c3 = b3c3_model(input_t1)
    t2_b3c3 = b3c3_model(input_t2)

    t1_b2c2 = b2c2_model(input_t1)
    t2_b2c2 = b2c2_model(input_t2)

    t1_b1c2 = b1c2_model(input_t1)
    t2_b1c2 = b1c2_model(input_t2)

    feature_left = vgg_model(input_t1)
    feature_right = vgg_model(input_t2)
    #自定义Lambda层，对两个特征相减
    diff = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    l1_distance = diff([feature_left, feature_right])

    conv_1 = Conv2D(512, (3, 3), activation='relu', padding='same')(l1_distance)
    up_conv = Conv2DTranspose(256, (3, 3), strides=(2, 2), activation='relu', padding='same')(conv_1)
    # first concatenation block
    concat_1 = concatenate([up_conv, t1_b5c3, t2_b5c3], axis=-1, name='concat_1')
    conv_2 = Conv2D(512, (3, 3), activation='relu', padding='same')(concat_1)
    up_conv_2 = Conv2DTranspose(256, (3, 3), strides=(2, 2), activation='relu', padding='same')(conv_2)

    # second concatenation block
    concat_2 = concatenate([up_conv_2, t1_b4c3, t2_b4c3], axis=-1, name='concat_2')
    conv_3 = Conv2D(512, (3, 3), activation='relu', padding='same')(concat_2)
    up_conv_3 = Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same')(conv_3)

    # third concatenation block
    concat_3 = concatenate([up_conv_3, t1_b3c3,t2_b3c3], axis=-1, name='concat_3')
    conv_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(concat_3)
    up_conv_4 = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(conv_4)

    # fourth concatenation block
    concat_4 = concatenate([up_conv_4, t1_b2c2, t2_b2c2], axis=-1, name='concat_4')
    conv_5 = Conv2D(128, (3, 3), activation='relu', padding='same')(concat_4)
    up_conv_5 = Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(conv_5)

    # fifth concatenation block
    concat_4 = concatenate([up_conv_5, t1_b1c2,t2_b1c2], axis=-1, name='concat_5')
    # conv_6 = Conv2D(128, (3, 3), activation='sigmoid', padding='same')(concat_4)
    x = Conv2D(
        filters=1,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv',
    )(concat_4)
    out = Activation('sigmoid', name='sigmoid')(x)

    final_model = Model([input_t1, input_t2], out)

    # w3= final_model.get_weights()
    final_model.summary()
    return final_model


if __name__=='__main__':

    # input_shape =(256,256,3)
    # model1 = siam_unet_vgg16(input_shape)
    # model2 = mynet_two(input_shape)

    print('ok')

