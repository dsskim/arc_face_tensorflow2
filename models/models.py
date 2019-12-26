import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dense,
    Dropout,
    Flatten,
    Input,
    Activation
)
from tensorflow.keras.applications import (
    MobileNetV2,
    ResNet50,
    Xception
)
from .loss_layers import (
    ArcFace
)


def _regularizer(weights_decay=5e-4):
    return tf.keras.regularizers.l2(weights_decay)


def vgg_block(x, filters, layers):
    for _ in range(layers):
        x = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal',
                                   kernel_regularizer=_regularizer())(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    return x


def vgg8(input_shape, name='vgg8'):
    input = Input(shape=input_shape)

    x = vgg_block(input, 16, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = vgg_block(x, 32, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = vgg_block(x, 64, 2)
    out = MaxPooling2D(pool_size=(2, 2))(x)

    return tf.keras.Model(input, out, name=name)


def Backbone(backbone_type='ResNet50', use_pretrain=True):
    """Backbone Model"""
    weights = None
    if use_pretrain:
        weights = 'imagenet'

    def backbone(x_in):
        if backbone_type == 'ResNet50':
            return ResNet50(input_shape=x_in.shape[1:], include_top=False,
                            weights=weights)(x_in)
        elif backbone_type == 'MobileNetV2':
            return MobileNetV2(input_shape=x_in.shape[1:], include_top=False,
                               weights=weights)(x_in)
        elif backbone_type == 'Xception':
            return Xception(input_shape=x_in.shape[1:], include_top=False,
                            weights=weights)(x_in)
        elif backbone_type == 'vgg8':
            return vgg8(input_shape=x_in.shape[1:])(x_in)
        else:
            raise TypeError('backbone_type error!')

    return backbone


def OutputLayer(embd_shape, name='OutputLayer'):
    """Output Later"""

    def output_layer(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = BatchNormalization()(x)
        x = Dropout(rate=0.5)(x)
        x = Flatten()(x)
        x = Dense(embd_shape, kernel_regularizer=_regularizer(), name='embedding')(x)
        x = BatchNormalization()(x)
        return Model(inputs, x, name=name)(x_in)

    return output_layer


def ArcFaceModel(size=None, channels=3, num_classes=None, name='arcface_model',
                 margin=0.5, logist_scale=64, embd_shape=512,
                 head_type='ArcHead', backbone_type='ResNet50',
                 use_pretrain=True, training=False):
    """Arc Face Model"""
    inputs = Input([size, size, channels], name='input_image')

    x = Backbone(backbone_type=backbone_type, use_pretrain=use_pretrain)(inputs)

    embds = OutputLayer(embd_shape)(x)

    if training:
        assert num_classes is not None
        labels = Input([], name='label')
        if head_type == 'ArcHead':
            logits = ArcFace(n_classes=num_classes, m=margin, s=logist_scale, regularizer=_regularizer())(
                [embds, labels])
        else:
            logits = Dense(num_classes, kernel_regularizer=_regularizer())(embds)
            logits = tf.nn.softmax(logits)
        return Model([inputs, labels], logits, name=name)
    else:
        return Model(inputs, embds, name=name)
