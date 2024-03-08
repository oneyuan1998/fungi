import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
import tensorflow_addons as tfa

class Mish(Layer):

    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

def convnext_block(x, dims):
    # residual = x
    x = ReflectionPadding2D((3,3))(x)
    x = DepthwiseConv2D(7, padding = 'valid', depthwise_initializer='he_normal', bias_initializer = 'zeros')(x)
    x = BatchNormalization()(x)
    x = Conv2D(dims*4, 1, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(x)
    x = tfa.layers.GELU()(x)
    x = Conv2D(dims, 1, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(x)
    x = BatchNormalization()(x)
    out = tfa.layers.GELU()(x)

    return out 

def conv2d_norm(x, filters, kernel_size=(3, 3), padding='same', groups=1, strides=(1, 1), activation=None, regularizer = None, norm = 'bn',name=None):

    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding, groups=groups,use_bias=True, kernel_initializer = 'he_normal', bias_initializer = 'zeros', kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
    if norm == 'bn':
        x = BatchNormalization()(x)
    elif norm == 'ln':
        x = LayerNormalization()(x)
    # x = BatchNormalization(axis = 3, scale = True)(x)

    if activation == 'mish':
        x = Mish()(x)
        return x
    elif activation == None:
        return x
    else:
        x = Activation(activation, name=name)(x)
        return x

def conv2d_bn(x, filters, kernel_size=(3, 3), padding='same', groups=1, strides=(1, 1), activation=None, regularizer = None, name=None):

    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding, groups=groups,use_bias=True, kernel_initializer = 'he_normal', bias_initializer = 'he_normal', kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
    
    x = BatchNormalization(axis=3, scale=False)(x)

    if activation == 'mish':
        x = Mish()(x)
        return x
    elif activation == None:
        return x
    else:
        x = Activation(activation, name=name)(x)
        return x

def DepthwiseConv2D_bn(x, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu', use_bias=True, regularizer = None, name=None):

    x = DepthwiseConv2D(kernel_size=kernel_size, strides = strides, padding=padding, activation=None, use_bias=use_bias, depthwise_initializer = 'he_normal', bias_initializer = 'zeros', kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)

    x = BatchNormalization()(x)

    if activation == 'mish':
        x = Mish()(x)
        return x
    elif activation == None:
        return x
    else:
        x = Activation(activation, name=name)(x)
        return x

def MultiResBlock(inp, U, activation = 'relu'):
    # U = U *1.67
    shortcut = inp
    shortcut = conv2d_norm(shortcut, filters=int(U*0.167) + int(U*0.333) + int(U*0.5), kernel_size=(1, 1), activation=None, padding='same')
    conv3x3 = conv2d_norm(inp, filters=int(U*0.167), kernel_size=(3, 3), activation=activation, padding='same')
    conv5x5 = conv2d_norm(conv3x3, filters=int(U*0.333), kernel_size=(3, 3), activation=activation, padding='same')
    conv7x7 = conv2d_norm(conv5x5, filters=int(U*0.5), kernel_size=(3, 3), activation=activation, padding='same')
    out = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    out = BatchNormalization()(out)
    out = add([shortcut, out])
    if activation == 'mish':
        out = Mish()(out)
    elif activation == None:
        pass
    else:
        out = Activation(activation)(out)
    out = BatchNormalization()(out)
    return out

def ResPath(filters, length, inp, activation = 'relu'):

    shortcut = inp
    shortcut = conv2d_norm(shortcut, filters=filters, kernel_size=(1, 1), activation=None, padding='same')

    out = conv2d_norm(inp, filters=filters, kernel_size=(3, 3), activation=activation, padding='same')

    out = add([shortcut, out])

    if activation == 'mish':
        out = Mish()(out)
    elif activation == None:
        pass
    else:
        out = Activation(activation)(out)

    out = BatchNormalization(axis=3)(out)

    for i in range(length-1):

        shortcut = out
        shortcut = conv2d_norm(shortcut, filters=filters, kernel_size=(1, 1), activation=None, padding='same')

        out = conv2d_norm(out, filters=filters, kernel_size=(3, 3), activation=activation, padding='same')

        out = add([shortcut, out])

        if activation == 'mish':
            out = Mish()(out)
        elif activation == None:
            pass
        else:
            out = Activation(activation)(out)
        
        out = BatchNormalization(axis=3)(out)

    return out

def Channel_wise_FE(x, filters, reshape_size = (2,2,2)):

    cw = GlobalAveragePooling2D()(x)
    cw = Dense(filters, activation='relu', kernel_initializer = 'he_normal', bias_initializer = 'he_normal')(cw)
    cw = Dense(filters, activation='sigmoid', kernel_initializer = 'he_normal', bias_initializer = 'he_normal')(cw)
    cw = Reshape(reshape_size)(cw)
    out = multiply([x,cw])

    return out

def Spatial_wise_FE(x, filters):
    sw = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True, kernel_initializer = 'he_normal', bias_initializer = 'he_normal')(x)
    sw = Activation('relu')(sw)
    sw = DepthwiseConv2D_bn(sw, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='sigmoid')
    out = multiply([x,sw])

    return out

def AFE(x, filters, reshape_size = (2,2,2)):

    Sr = x
    Sc = Channel_wise_FE(Sr, filters, reshape_size = reshape_size)
    Ss = Spatial_wise_FE(Sr, filters)
    So = concatenate([Sr,Sc,Ss], axis = 3)
    So = Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True, kernel_initializer = 'he_normal', bias_initializer = 'he_normal')(So)
    So = Mish()(So)
    So = add([x, So])
    So = BatchNormalization()(So)

    return So

def EF3_Net(pretrained_weights = None,input_size = (256,256,1)):
    kn = 32
    km1 = int(kn*0.167) +    int(kn*0.333) +    int(kn*0.5)
    km2 = int(kn*2*0.167) +  int(kn*2*0.333) +  int(kn*2*0.5)
    km3 = int(kn*4*0.167) +  int(kn*4*0.333) +  int(kn*4*0.5)
    km4 = int(kn*8*0.167) +  int(kn*8*0.333) +  int(kn*8*0.5)
    km5 = int(kn*14*0.167) + int(kn*14*0.333) + int(kn*14*0.5)


    size0 = input_size[0]
    size1 = input_size[1]
    size2 = input_size[2]

    inputs = Input(input_size)

    mresblock1 = MultiResBlock(inputs, kn)
    pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)
    mresblock1 = Spatial_wise_FE(mresblock1, km1)
    mresblock1 = BatchNormalization(axis=-1, center=True, scale=False)(mresblock1)

    mresblock2 = MultiResBlock(pool1, kn*2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
    mresblock2 = AFE(mresblock2, km2, reshape_size = (1, 1, km2))

    mresblock3 = MultiResBlock(pool2, kn*4)
    pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
    mresblock3 = AFE(mresblock3, km3, reshape_size = (1, 1, km3))

    mresblock4 = MultiResBlock(pool3, kn*8)
    pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)
    mresblock4 = Channel_wise_FE(mresblock4, km4, reshape_size = (1, 1, km4))
    mresblock4 = BatchNormalization(axis=-1, scale=False)(mresblock4)

    mresblock5 = MultiResBlock(pool4, kn*14)
    mresblock5_ccn = Channel_wise_FE(mresblock5, km5, reshape_size = (1, 1, km5))
    mresblock5_ccn = BatchNormalization(axis=-1, scale=False)(mresblock5_ccn)

    up6 = concatenate([Conv2DTranspose(kn*8, (2, 2), strides=(2, 2), padding='same')(mresblock5), mresblock4], axis=3)
    mresblock6 = MultiResBlock(up6, kn*8)
    mresblock6_ccn = Channel_wise_FE(mresblock6, km4, reshape_size = (1, 1, km4))
    mresblock6_ccn = BatchNormalization(axis=-1, scale=False)(mresblock6_ccn)

    up7 = concatenate([Conv2DTranspose(kn*4, (2, 2), strides=(2, 2), padding='same')(mresblock6), mresblock3], axis=3)
    mresblock7 = MultiResBlock(up7, kn*4)
    mresblock7_ccn = AFE(mresblock7, km3, reshape_size = (1, 1, km3))

    up8 = concatenate([Conv2DTranspose(kn*2, (2, 2), strides=(2, 2), padding='same')(mresblock7), mresblock2], axis=3)
    mresblock8 = MultiResBlock(up8, kn*2)
    mresblock8_ccn = AFE(mresblock8, km2, reshape_size = (1, 1, km2))

    up9 = concatenate([Conv2DTranspose(kn, (2, 2), strides=(2, 2), padding='same')(mresblock8), mresblock1], axis=3)
    mresblock9 = MultiResBlock(up9, kn)
    mresblock9 = conv2d_norm(mresblock9, 3, (1, 1), activation='relu')



    inputs2 = concatenate([inputs, mresblock9], axis=3)
    mresblock10 = MultiResBlock(inputs2, kn)
    pool10 = MaxPooling2D(pool_size=(2, 2))(mresblock10)
    mresblock10 = Spatial_wise_FE(mresblock10, km1)
    mresblock10 = BatchNormalization(axis=-1, scale=False)(mresblock10)
    merge10 = concatenate([mresblock8_ccn, pool10], axis=3)
    
    mresblock11 = MultiResBlock(merge10, kn*2)
    pool11 = MaxPooling2D(pool_size=(2, 2))(mresblock11)
    mresblock11 = AFE(mresblock11, km2, reshape_size = (1, 1, km2))
    merge11 = concatenate([mresblock7_ccn, pool11], axis=3)

    mresblock12 = MultiResBlock(merge11, kn*4)
    pool12 = MaxPooling2D(pool_size=(2, 2))(mresblock12)
    mresblock12 = AFE(mresblock12, km3, reshape_size = (1, 1, km3))
    merge12 = concatenate([mresblock6_ccn, pool12], axis=3)

    mresblock13 = MultiResBlock(merge12, kn*8)
    pool13 = MaxPooling2D(pool_size=(2, 2))(mresblock13)
    mresblock13 = Channel_wise_FE(mresblock13, km4, reshape_size = (1, 1, km4))
    mresblock13 = BatchNormalization(axis=-1, scale=False)(mresblock13)
    merge13 = concatenate([mresblock5_ccn, pool13], axis=3)

    mresblock14 = MultiResBlock(merge13, kn*16)

    up15 = concatenate([Conv2DTranspose(kn*8, (2, 2), strides=(2, 2), padding='same')(mresblock14), mresblock13], axis=3)
    mresblock15 = MultiResBlock(up15, kn*8)

    up16 = concatenate([Conv2DTranspose(kn*4, (2, 2), strides=(2, 2), padding='same')(mresblock15), mresblock12], axis=3)
    mresblock16 = MultiResBlock(up16, kn*4)

    up17 = concatenate([Conv2DTranspose(kn*2, (2, 2), strides=(2, 2), padding='same')(mresblock16), mresblock11], axis=3)
    mresblock17 = MultiResBlock(up17, kn*2)

    up18 = concatenate([Conv2DTranspose(kn,   (2, 2), strides=(2, 2), padding='same')(mresblock17), mresblock10], axis=3)
    mresblock18 = MultiResBlock(up18, kn)

    
    
    conv19 = conv2d_norm(mresblock18, 3, (1, 1), activation='softmax')

    model = Model(inputs = inputs, outputs = conv19)
    
    # model.summary()
    

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def EF_MRUNet(pretrained_weights = None,input_size = (256,256,1)):
    kn = 50
    km1 = int(kn*0.167) + int(kn*0.333) + int(kn*0.5)
    km2 = int(kn*2*0.167) + int(kn*2*0.333) + int(kn*2*0.5)
    km3 = int(kn*4*0.167) + int(kn*4*0.333) + int(kn*4*0.5)

    inputs = Input(input_size)

    mresblock1 = MultiResBlock(inputs, kn)
    pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)
    mresblock1 = Spatial_wise_FE(mresblock1, km1)
    mresblock1 = BatchNormalization(axis=3, center=True, scale=False)(mresblock1)

    mresblock2 = MultiResBlock(pool1, kn*2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
    mresblock2 = AFE(mresblock2, km2, reshape_size = (1, 1, km2))

    mresblock3 = MultiResBlock(pool2, kn*4)
    pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
    mresblock3 = AFE(mresblock3, km3, reshape_size = (1, 1, km3))

    mresblock4 = MultiResBlock(pool3, kn*8)
    pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)
    mresblock4 = Channel_wise_FE(mresblock4, kn*8, reshape_size = (1, 1, kn*8))
    mresblock4 = BatchNormalization(axis=3, center=True, scale=False)(mresblock4)

    mresblock5 = MultiResBlock(pool4, kn*16)

    up6 = concatenate([Conv2DTranspose(kn*8, (2, 2), strides=(2, 2), padding='same')(mresblock5), mresblock4], axis=3)
    mresblock6 = MultiResBlock(up6, kn*8)

    up7 = concatenate([Conv2DTranspose(kn*4, (2, 2), strides=(2, 2), padding='same')(mresblock6), mresblock3], axis=3)
    mresblock7 = MultiResBlock(up7, kn*4)

    up8 = concatenate([Conv2DTranspose(kn*2, (2, 2), strides=(2, 2), padding='same')(mresblock7), mresblock2], axis=3)
    mresblock8 = MultiResBlock(up8, kn*2)

    up9 = concatenate([Conv2DTranspose(kn, (2, 2), strides=(2, 2), padding='same')(mresblock8), mresblock1], axis=3)
    mresblock9 = MultiResBlock(up9, kn)

    conv10 = conv2d_norm(mresblock9, 1, (1, 1), activation='sigmoid')
    
    model = Model(inputs = inputs, outputs = conv10)
    
    model.summary()
    

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def MultiResUNet(pretrained_weights = None ,input_size = (256,256,1), activation = 'relu'):
    kn = 32

    inputs = Input(input_size)

    mresblock1 = MultiResBlock(inputs, kn)
    pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)
    mresblock1 = ResPath(kn, 4, mresblock1)

    mresblock2 = MultiResBlock(pool1, kn*2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
    mresblock2 = ResPath(kn*2, 3, mresblock2)

    mresblock3 = MultiResBlock(pool2, kn*4)
    pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
    mresblock3 = ResPath(kn*4, 2, mresblock3)

    mresblock4 = MultiResBlock(pool3, kn*8)
    pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)
    mresblock4 = ResPath(kn*8, 1, mresblock4)

    mresblock5 = MultiResBlock(pool4, kn*16)

    up6 = Conv2DTranspose(kn*8, (2, 2), strides=(2, 2), padding='same')(mresblock5)
    up6 = concatenate([up6, mresblock4], axis=3)
    mresblock6 = MultiResBlock(up6, kn*8)

    up7 = Conv2DTranspose(kn*4, (2, 2), strides=(2, 2), padding='same')(mresblock6)
    up7 = concatenate([up7, mresblock3], axis=3)
    mresblock7 = MultiResBlock(up7, kn*4)

    up8 = Conv2DTranspose(kn*2, (2, 2), strides=(2, 2), padding='same')(mresblock7)
    up8 = concatenate([up8, mresblock2], axis=3)
    mresblock8 = MultiResBlock(up8, kn*2)

    up9 = Conv2DTranspose(kn, (2, 2), strides=(2, 2), padding='same')(mresblock8)
    up9 = concatenate([up9, mresblock1], axis=3)
    mresblock9 = MultiResBlock(up9, kn)
    conv9 = conv2d_norm(mresblock9, 3, (1, 1), activation = 'softmax')

    model = Model(inputs = inputs, outputs = conv9)
    # model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def unet(pretrained_weights = None,input_size = (256,256,1)):
    kn=32

    inputs = Input(input_size)
    conv1 = Conv2D(kn, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(kn, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(kn*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(kn*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(kn*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(kn*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(kn*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(kn*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    
    

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(kn*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(kn*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)


    up6 = Conv2D(kn*8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    merge6 = concatenate([conv4,up6], axis = 3)
    conv6 = Conv2D(kn*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(kn*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(kn*4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(kn*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(kn*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(kn*2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(kn*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(kn*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(kn, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(kn, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(kn, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    # conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(3, 1, activation = 'softmax')(conv9)
    
    model = Model(inputs = inputs, outputs = conv10)
    # model.summary()

    
    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def unet_bn(pretrained_weights = None,input_size = (256,256,1)):
    kn=32

    inputs = Input(input_size)
    conv1 = conv2d_norm(inputs, kn, 3,'same',activation='relu')
    conv1 = conv2d_norm(conv1, kn, 3,'same',activation='relu')
 
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv2d_norm(pool1, kn*2, activation='relu')
    conv2 = conv2d_norm(conv2, kn*2, activation='relu')
 
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv2d_norm(pool2, kn*4, activation='relu')
    conv3 = conv2d_norm(conv3, kn*4, activation='relu')
 
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv2d_norm(pool3, kn*8, activation='relu')
    conv4 = conv2d_norm(conv4, kn*8, activation='relu')
 
    
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = conv2d_norm(pool4, kn*16, activation='relu')
    conv5 = conv2d_norm(conv5, kn*16, activation='relu')



    up6 = Conv2D(kn*8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    merge6 = concatenate([conv4,up6], axis = 3)
    conv6 = conv2d_norm(merge6, kn*8, activation='relu')
    conv6 = conv2d_norm(conv6, kn*8, activation='relu')

    up7 = Conv2D(kn*4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = conv2d_norm(merge7, kn*4, activation='relu')
    conv7 = conv2d_norm(conv7, kn*4, activation='relu')

    up8 = Conv2D(kn*2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = conv2d_norm(merge8, kn*2, activation='relu')
    conv8 = conv2d_norm(conv8, kn*2, activation='relu')

    up9 = Conv2D(kn, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = conv2d_norm(merge9, kn, activation='relu')
    conv9 = conv2d_norm(conv9, kn, activation='relu')
    # conv9 = conv2d_norm(conv9, 2, activation='relu')
    conv9 = conv2d_norm(conv9, 3, activation='softmax')
    
    # conv9 = conv2d_norm(conv9, 1, activation='sigmoid')

    model = Model(inputs = inputs, outputs = conv9)

    if(pretrained_weights):
        model.load_weights(pretrained_weights)
    return model



# resunet
def bn_act(x, act=True):
    'batch normalization layer with an optinal activation layer'
    x = tf.keras.layers.BatchNormalization()(x)
    if act == True:
        x = tf.keras.layers.Activation('relu')(x)
    return x

def conv_block(x, filters, kernel_size=3, padding='same', strides=1):
    'convolutional layer which always uses the batch normalization layer'
    conv = bn_act(x)
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv


def stem(x, filters, kernel_size=3, padding='same', strides=1):
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size, padding, strides)
    shortcut = Conv2D(filters, kernel_size=1, padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    output = Add()([conv, shortcut])
    return output

def residual_block(x, filters, kernel_size=3, padding='same', strides=1):
    res = conv_block(x, filters, kernel_size, padding, strides)
    res = conv_block(res, filters, kernel_size, padding, 1)
    shortcut = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    output = Add()([shortcut, res])
    return output

def upsample_concat_block(x, xskip):
    u = UpSampling2D((2,2))(x)
    c = Concatenate()([u, xskip])
    return c

def ResUNet(pretrained_weights = None,input_size = (256,256,1)):
    f = [16, 32, 64, 128, 256]
    inputs = Input(input_size)
    
    ## Encoder
    e0 = inputs
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    e5 = residual_block(e4, f[4], strides=2)
    
    ## Bridge
    b0 = conv_block(e5, f[4], strides=1)
    b1 = conv_block(b0, f[4], strides=1)
    
    ## Decoder
    u1 = upsample_concat_block(b1, e4)
    d1 = residual_block(u1, f[4])
    
    u2 = upsample_concat_block(d1, e3)
    d2 = residual_block(u2, f[3])
    
    u3 = upsample_concat_block(d2, e2)
    d3 = residual_block(u3, f[2])
    
    u4 = upsample_concat_block(d3, e1)
    d4 = residual_block(u4, f[1])
    
    outputs = tf.keras.layers.Conv2D(3, (1, 1), padding="same", activation="softmax")(d4)
    model = tf.keras.models.Model(inputs, outputs)
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
    return model

# SegNet
class MaxPoolingWithArgmax2D(Layer):

    def __init__(
            self,
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same',
            **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        if K.backend() == 'tensorflow':
            ksize = [1, pool_size[0], pool_size[1], 1]
            padding = padding.upper()
            strides = [1, strides[0], strides[1], 1]
            output, argmax = tf.nn.max_pool_with_argmax(
                    inputs,
                    ksize=ksize,
                    strides=strides,
                    padding=padding)
        else:
            errmsg = '{} backend is not supported for layer {}'.format(
                    K.backend(), type(self).__name__)
            raise NotImplementedError(errmsg)
        argmax = K.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [
                dim//ratio[idx]
                if dim is not None else None
                for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]

class MaxUnpooling2D(Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        with tf.compat.v1.variable_scope(self.name):
            mask = K.cast(mask, 'int32')
            input_shape = tf.shape(updates, out_type='int32')
            #print(updates.shape)
            #print(mask.shape)
            if output_shape is None:
                output_shape = (
                    input_shape[0],
                    input_shape[1] * self.size[0],
                    input_shape[2] * self.size[1],
                    input_shape[3])

            ret = tf.scatter_nd(K.expand_dims(K.flatten(mask)),
                                K.flatten(updates),
                                [K.prod(output_shape)])

            input_shape = updates.shape
            out_shape = [-1,
                        input_shape[1] * self.size[0],
                        input_shape[2] * self.size[1],
                        input_shape[3]]
        return K.reshape(ret, out_shape)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'size': self.size
        })
        return config

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return (
                mask_shape[0],
                mask_shape[1]*self.size[0],
                mask_shape[2]*self.size[1],
                mask_shape[3]
                )

def SegNet_ori(pretrained_weights = None, input_size = (1024, 1024, 1), n_labels = 1, kernel=3, pool_size=(2, 2)):
    
    inputs = Input(shape=input_size)

    conv_1 = Convolution2D(64, (kernel, kernel), padding="same")(inputs)
    conv_1 = BatchNormalization(scale=False)(conv_1)
    conv_1 = Activation("relu")(conv_1)
    conv_2 = Convolution2D(64, (kernel, kernel), padding="same")(conv_1)
    conv_2 = BatchNormalization(scale=False)(conv_2)
    conv_2 = Activation("relu")(conv_2)

    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)

    conv_3 = Convolution2D(128, (kernel, kernel), padding="same")(pool_1)
    conv_3 = BatchNormalization(scale=False)(conv_3)
    conv_3 = Activation("relu")(conv_3)
    conv_4 = Convolution2D(128, (kernel, kernel), padding="same")(conv_3)
    conv_4 = BatchNormalization(scale=False)(conv_4)
    conv_4 = Activation("relu")(conv_4)

    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

    conv_5 = Convolution2D(256, (kernel, kernel), padding="same")(pool_2)
    conv_5 = BatchNormalization(scale=False)(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_6 = Convolution2D(256, (kernel, kernel), padding="same")(conv_5)
    conv_6 = BatchNormalization(scale=False)(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_7 = Convolution2D(256, (kernel, kernel), padding="same")(conv_6)
    conv_7 = BatchNormalization(scale=False)(conv_7)
    conv_7 = Activation("relu")(conv_7)

    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)

    conv_8 = Convolution2D(512, (kernel, kernel), padding="same")(pool_3)
    conv_8 = BatchNormalization(scale=False)(conv_8)
    conv_8 = Activation("relu")(conv_8)
    conv_9 = Convolution2D(512, (kernel, kernel), padding="same")(conv_8)
    conv_9 = BatchNormalization(scale=False)(conv_9)
    conv_9 = Activation("relu")(conv_9)
    conv_10 = Convolution2D(512, (kernel, kernel), padding="same")(conv_9)
    conv_10 = BatchNormalization(scale=False)(conv_10)
    conv_10 = Activation("relu")(conv_10)

    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)

    conv_11 = Convolution2D(512, (kernel, kernel), padding="same")(pool_4)
    conv_11 = BatchNormalization(scale=False)(conv_11)
    conv_11 = Activation("relu")(conv_11)
    conv_12 = Convolution2D(512, (kernel, kernel), padding="same")(conv_11)
    conv_12 = BatchNormalization(scale=False)(conv_12)
    conv_12 = Activation("relu")(conv_12)
    conv_13 = Convolution2D(512, (kernel, kernel), padding="same")(conv_12)
    conv_13 = BatchNormalization(scale=False)(conv_13)
    conv_13 = Activation("relu")(conv_13)

    pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)
    

    unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])

    conv_14 = Convolution2D(512, (kernel, kernel), padding="same")(unpool_1)
    conv_14 = BatchNormalization(scale=False)(conv_14)
    conv_14 = Activation("relu")(conv_14)
    conv_15 = Convolution2D(512, (kernel, kernel), padding="same")(conv_14)
    conv_15 = BatchNormalization(scale=False)(conv_15)
    conv_15 = Activation("relu")(conv_15)
    conv_16 = Convolution2D(512, (kernel, kernel), padding="same")(conv_15)
    conv_16 = BatchNormalization(scale=False)(conv_16)
    conv_16 = Activation("relu")(conv_16)

    unpool_2 = MaxUnpooling2D(pool_size)([conv_16, mask_4])

    conv_17 = Convolution2D(512, (kernel, kernel), padding="same")(unpool_2)
    conv_17 = BatchNormalization(scale=False)(conv_17)
    conv_17 = Activation("relu")(conv_17)
    conv_18 = Convolution2D(512, (kernel, kernel), padding="same")(conv_17)
    conv_18 = BatchNormalization(scale=False)(conv_18)
    conv_18 = Activation("relu")(conv_18)
    conv_19 = Convolution2D(256, (kernel, kernel), padding="same")(conv_18)
    conv_19 = BatchNormalization(scale=False)(conv_19)
    conv_19 = Activation("relu")(conv_19)

    unpool_3 = MaxUnpooling2D(pool_size)([conv_19, mask_3])

    conv_20 = Convolution2D(256, (kernel, kernel), padding="same")(unpool_3)
    conv_20 = BatchNormalization(scale=False)(conv_20)
    conv_20 = Activation("relu")(conv_20)
    conv_21 = Convolution2D(256, (kernel, kernel), padding="same")(conv_20)
    conv_21 = BatchNormalization(scale=False)(conv_21)
    conv_21 = Activation("relu")(conv_21)
    conv_22 = Convolution2D(128, (kernel, kernel), padding="same")(conv_21)
    conv_22 = BatchNormalization(scale=False)(conv_22)
    conv_22 = Activation("relu")(conv_22)

    unpool_4 = MaxUnpooling2D(pool_size)([conv_22, mask_2])

    conv_23 = Convolution2D(128, (kernel, kernel), padding="same")(unpool_4)
    conv_23 = BatchNormalization(scale=False)(conv_23)
    conv_23 = Activation("relu")(conv_23)
    conv_24 = Convolution2D(64, (kernel, kernel), padding="same")(conv_23)
    conv_24 = BatchNormalization(scale=False)(conv_24)
    conv_24 = Activation("relu")(conv_24)

    unpool_5 = MaxUnpooling2D(pool_size)([conv_24, mask_1])

    conv_25 = Convolution2D(64, (kernel, kernel), padding="same")(unpool_5)
    conv_25 = BatchNormalization(scale=False)(conv_25)
    conv_25 = Activation("relu")(conv_25)

    conv_26 = Convolution2D(n_labels, (1, 1), padding="valid")(conv_25)
    conv_26 = BatchNormalization(scale=False)(conv_26)
    conv_26 = Activation("softmax")(conv_26)
    
    
    model = Model(inputs = inputs, outputs = conv_26)
    
    # model.summary()
    
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
    
    return model
