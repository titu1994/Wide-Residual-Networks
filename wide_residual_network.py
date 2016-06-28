from keras.layers import merge, Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization

def initial_conv(input, init_filter=16):
    x = Convolution2D(init_filter, 3, 3, border_mode='same')(input)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    return x

def conv1_block(input, k=1, dropout=0.0):
    init = input

    # Check if input number of filters is same as 16 * k, else create convolution2d for this input
    if init._keras_shape[1] != 16 * k:
        init = Convolution2D(16 * k, 1, 1, activation='linear', border_mode='same')(init)

    x = Convolution2D(16 * k, 3, 3, border_mode='same')(input)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = Convolution2D(16 * k, 3, 3, border_mode='same')(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    m = merge([init, x], mode='sum')
    return m

def conv2_block(input, k=1, dropout=0.0):
    init = input

    # Check if input number of filters is same as 16 * k, else create convolution2d for this input
    if init._keras_shape[1] != 32 * k:
        init = Convolution2D(32 * k, 1, 1, activation='linear', border_mode='same')(init)

    x = Convolution2D(32 * k, 3, 3, border_mode='same')(input)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = Convolution2D(32 * k, 3, 3, border_mode='same')(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    m = merge([init, x], mode='sum')
    return m

def conv3_block(input, k=1, dropout=0.0):
    init = input

    # Check if input number of filters is same as 16 * k, else create convolution2d for this input
    if init._keras_shape[1] != 64 * k:
        init = Convolution2D(64 * k, 1, 1, activation='linear', border_mode='same')(init)

    x = Convolution2D(64 * k, 3, 3, border_mode='same')(input)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = Convolution2D(64 * k, 3, 3, border_mode='same')(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    m = merge([init, x], mode='sum')
    return m

def create_wide_residual_network(input, nb_classes=100, init_filter=16, N=2, k=1, dropout=0.0, verbose=1):
    x = initial_conv(input, init_filter)
    nb_conv = 4

    for i in range(N):
        x = conv1_block(x, k, dropout)
        nb_conv += 2

    x = MaxPooling2D((2,2))(x)

    for i in range(N):
        x = conv2_block(x, k, dropout)
        nb_conv += 2

    x = MaxPooling2D((2,2))(x)

    for i in range(N):
        x = conv3_block(x, k, dropout)
        nb_conv += 2

    x = AveragePooling2D((8,8))(x)
    x = Flatten()(x)

    x = Dense(nb_classes, activation='softmax')(x)

    if verbose: print("Wide Residual Network-%d-%d created." % (nb_conv, k))
    return x

if __name__ == "__main__":
    from keras.utils.visualize_util import plot
    from keras.layers import Input
    from keras.models import Model

    init = Input(shape=(3, 32, 32))

    wrn_28_10 = create_wide_residual_network(init, nb_classes=100, N=4, k=10, dropout=0.25)

    model = Model(init, wrn_28_10)

    model.summary()
    plot(model, "WRN-28-10.png", show_shapes=True, show_layer_names=True)