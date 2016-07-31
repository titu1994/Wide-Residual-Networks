import numpy as np
import sklearn.metrics as metrics

import wide_residual_network as wrn
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input
import keras.callbacks as callbacks
import keras.utils.np_utils as kutils
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.visualize_util import plot

batch_size = 64
nb_epoch = 100
img_rows, img_cols = 32, 32

(trainX, trainY), (testX, testY) = cifar10.load_data()

trainX = trainX.astype('float32')
trainX /= 255.0
testX = testX.astype('float32')
testX /= 255.0

tempY = testY
trainY = kutils.to_categorical(trainY)
testY = kutils.to_categorical(testY)

generator = ImageDataGenerator(featurewise_center=True,
                               featurewise_std_normalization=True,
                               rotation_range=10,
                               width_shift_range=5./32,
                               height_shift_range=5./32,
                               horizontal_flip=True)

generator.fit(trainX, seed=0, augment=True)

test_generator = ImageDataGenerator(featurewise_center=True,
                                    featurewise_std_normalization=True,)

test_generator.fit(testX, seed=0, augment=True)

init = Input(shape=(3, img_rows, img_cols),)

# For WRN-16-8 put N = 2, k = 8
# For WRN-28-10 put N = 4, k = 10
# For WRN-40-4 put N = 9, k = 4
wrn_28_8 = wrn.create_wide_residual_network(init, nb_classes=10, N=4, k=8, dropout=0.0)

model = Model(input=init, output=wrn_28_8)

model.summary()
plot(model, "WRN-28-8.png", show_shapes=False)

model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["acc"])
print("Finished compiling")
print("Allocating GPU memory")

model.load_weights("WRN-28-8 Weights.h5")
print("Model loaded.")

#model.fit_generator(generator.flow(trainX, trainY, batch_size=batch_size), samples_per_epoch=len(trainX), nb_epoch=nb_epoch,
#                    callbacks=[callbacks.ModelCheckpoint("WRN-28-8 Weights.h5", monitor="val_acc", save_best_only=True)],
#                    validation_data=test_generator.flow(testX, testY, batch_size=batch_size),
#                    nb_val_samples=testX.shape[0],)

scores = model.evaluate_generator(test_generator.flow(testX, testY, nb_epoch), testX.shape[0])
print("Accuracy = %f" % (100 * scores[1]))