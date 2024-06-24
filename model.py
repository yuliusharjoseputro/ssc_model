from config import get_config
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, GlobalAveragePooling2D, Dropout, Flatten, Dense, Multiply, Lambda
from tensorflow.keras.layers import AveragePooling2D, UpSampling2D, Reshape, add, MaxPooling2D
from tensorflow.keras.models import Model

def batchnorm_relu(inputs):
    """ Batch Normalization & ReLU """
    x = BatchNormalization()(inputs)
    x = Activation("relu")(x)
    x = Dropout(0.2)(x)
    return x

def residual_block(inputs, num_filters, strides=1):
    """ Convolutional Layers """
    x = batchnorm_relu(inputs)
    x = Conv2D(num_filters, 3, padding="same", strides=strides)(x)
    x = batchnorm_relu(x)
    x = Conv2D(num_filters, 3, padding="same", strides=1)(x)

    """ Shortcut Connection (Identity Mapping) """
    s = Conv2D(num_filters, 1, padding="same", strides=strides)(inputs)

    """ Addition """
    x = x + s
    return x

def decoder_block(inputs, skip_features, num_filters):
    """ Decoder Block """

    x = UpSampling2D((2, 2))(inputs)
    x = Concatenate()([x, skip_features])
    x = residual_block(x, num_filters, strides=1)
    return x

def SqueezeAndExcite(inputs, ratio=16):
    init = inputs
    filters = init.shape[-1]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    x = init * se
    return x

def ASPP(inputs, filters):
    shape = inputs.shape

    y_6 = Conv2D(filters=filters, kernel_size=3, dilation_rate=6, padding='same')(inputs)
    y_6 = BatchNormalization()(y_6)
    y_6 = Dropout(0.3)(y_6)
    y_6 = Activation('relu')(y_6)

    y_12 = Conv2D(filters=filters, kernel_size=3, dilation_rate=12, padding='same')(inputs)
    y_12 = BatchNormalization()(y_12)
    y_12 = Dropout(0.3)(y_12)
    y_12 = Activation('relu')(y_12)

    y_18 = Conv2D(filters=filters, kernel_size=3, dilation_rate=18, padding='same')(inputs)
    y_18 = BatchNormalization()(y_18)
    y_18 = Dropout(0.3)(y_18)
    y_18 = Activation('relu')(y_18)

    y = Concatenate()([y_6, y_12, y_18])
    # y = Concatenate()([y_6, y_12, y_18])

    y = Conv2D(filters=filters, kernel_size=1, padding='same')(y)
    # y = BatchNormalization()(y)
    # y = Activation('relu')(y)
    return y

def conv_block_trad(inputs, num_filters, strides=1):
    """ Convolutional Layers """
    x = batchnorm_relu(inputs)
    x = Conv2D(num_filters, 3, padding="same", strides=strides)(x)
    x = batchnorm_relu(x)
    x = Conv2D(num_filters, 3, padding="same", strides=strides)(x)

    return x

def decoder_block_trad(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block_trad(x, num_filters)
    return x

def SSC(cfg, dropout_rate):
    
    input_tensor = cfg["inputs"]
    num_class = cfg["num_classes_segmentation"]
    
    """ Endoder 1 """
    x = Conv2D(64, 3, padding="same", strides=1)(input_tensor)
    x = batchnorm_relu(x)
    x = Conv2D(64, 3, padding="same", strides=1)(x)
    s = Conv2D(64, 1, padding="same")(input_tensor)
    s1 = x + s
    s1 = SqueezeAndExcite(s1)
    p1 = MaxPool2D((2, 2))(s1)

    """ Encoder 2, 3, 4 """
    s2 = residual_block(p1, 128, strides=1)
    s2 = SqueezeAndExcite(s2)
    p2 = MaxPool2D((2, 2))(s2)

    s3 = residual_block(p2, 256, strides=1)
    s3 = SqueezeAndExcite(s3)
    p3 = MaxPool2D((2, 2))(s3)

    s4 = residual_block(p3, 512, strides=1)
    sa4 = SqueezeAndExcite(s4)
    p4 = MaxPool2D((2, 2))(sa4)

    """ Bridge """
    b1 = ASPP(p4, 1024)
    
    """ Decoder 1, 2, 3 """
    x = decoder_block(b1, s4, 512)
    x = decoder_block(x, s3, 256)
    x = decoder_block(x, s2, 128)
    x = decoder_block(x, s1, 64)

    b2 = ASPP(x, 64)

    """ Classifier """
    segmentation_output = Conv2D(num_class, (1,1), padding="same", activation="sigmoid", name="segmentation_output_hybrid")(b2)

    x = GlobalAveragePooling2D()(b1)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(dropout_rate)(x)

    classification_output = Dense(cfg["num_classes"], activation='softmax', name='model_classification')(x)

    model = Model(inputs= input_tensor, outputs=[segmentation_output, classification_output])

    return model


if __name__ == "__main__":

    """Load Config Hyperparameter"""
    cfg = get_config()

    combined_model = SSC(cfg, 0.3)
    combined_model.summary()

    