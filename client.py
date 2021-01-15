import tensorflow as tf
import numpy as np
import requests
import pickle

(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

NUM_IMAGES_TRAIN = 5000
NUM_CLASSES = 10
IMAGE_SHAPE = (28, 28)
url = 'http://127.0.0.1:5000/'


def trainOnData(model, trainData):

    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(0.001),
    #     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    #     metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    # )

    (X, y) = trainData

    model.fit(
        X,
        y,
        epochs=20
    )


if __name__ == '__main__':
    data = (X_train[:NUM_IMAGES_TRAIN], Y_train[:NUM_IMAGES_TRAIN])
    r = requests.get(url, allow_redirects=True)
    open('client.h5', 'wb').write(r.content)

    model = tf.keras.models.load_model('client.h5')
    trainOnData(model, data)
    model.save('client.h5')

    r = requests.post(url, files={'model': 'client.h5'})

    print(r.content)

