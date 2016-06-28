import numpy as np
import sklearn.metrics as metrics

import wide_residual_network as wrn
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input
import keras.callbacks as callbacks
from keras.callbacks import Callback
import keras.utils.np_utils as kutils
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.visualize_util import plot

class HistoryCheckpoint(Callback):
    '''Callback that records events
        into a `History` object after every epoch.
    '''

    def __init__(self, filename):
        super(Callback, self).__init__()
        self.filename = filename

    def on_train_begin(self, logs={}):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs={}):
        self.epoch.append(epoch)
        for k, v in logs.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(v)

        with open(self.filename, "w") as f:
            f.write(str(self.history))

batch_size = 128
nb_epoch = 100
img_rows, img_cols = 32, 32

(trainX, trainY), (testX, testY) = cifar10.load_data()

trainX = trainX.astype('float32')
trainX /= 255.0
validationX = testX.astype('float32')
validationX /= 255.0

trainY = kutils.to_categorical(trainY, 10)
validationY = kutils.to_categorical(testY, 10)

generator = ImageDataGenerator(rotation_range=10,
                               width_shift_range=5./32,
                               height_shift_range=5./32,
                               horizontal_flip=True)

generator.fit(trainX, seed=0)

init = Input(shape=(3, img_rows, img_cols),)

wrn_16_8 = wrn.create_wide_residual_network(init, nb_classes=10, N=2, k=8, dropout=0.01)

model = Model(input=init, output=wrn_16_8)

model.summary()
#plot(model, "WRN-16-8.png", show_shapes=False)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
print("Finished compiling")
print("Allocating GPU memory")

model.load_weights("WRN-16-8 Weights.h5")
print("Model loaded.")

#model.fit_generator(generator.flow(trainX, trainY, batch_size=batch_size), samples_per_epoch=len(trainX), nb_epoch=nb_epoch,
#                    callbacks=[callbacks.ModelCheckpoint("WRN-16-8 Weights.h5", monitor="val_acc", save_best_only=True),
#                               HistoryCheckpoint("WRN-16-8 History.txt"),],
#                    validation_data=(validationX, validationY))

yPreds = model.predict(validationX)
yPred = np.argmax(yPreds, axis=1)
yTrue = testY

accuracy = metrics.accuracy_score(yTrue, yPred) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)

import heapq
errorCount = 0
top5Correct = 0

for i in range(len(yPreds)):
    res = heapq.nlargest(5, range(len(yPreds[i])), yPreds[i].take)
    if yTrue[i] != res[0]:
        errorCount += 1

        if yTrue[i] in res[1:]:
            top5Correct += 1

print("Error count : ", errorCount)
print("Top 5 Guess Correct count : ", top5Correct)

print("Top 5 error count : %d" % (errorCount - top5Correct))