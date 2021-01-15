import tensorflow as tf
import numpy as np

(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()


NUM_IMAGES_TRAIN = 5000
NUM_CLASSES = 10
IMAGE_SHAPE = (28, 28)


def getNewModel():

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Dense(200))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Dense(200))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Dense(NUM_CLASSES))
    model.add(tf.keras.layers.Activation("softmax"))

    model.build(input_shape=(NUM_IMAGES_TRAIN, IMAGE_SHAPE[0], IMAGE_SHAPE[1]))

    return model


def updateParam(paramList):

    numModels = len(paramList)
    resParam = [layerWeight / numModels for layerWeight in paramList[0]]

    for param in paramList[1:]:
        for i, layerWeight in enumerate(param):
            resParam[i] += layerWeight / numModels

    return resParam


def predict(model, predData):

    return model.predict(predData)