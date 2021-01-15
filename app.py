from flask import Flask, send_file, request
import os
from utils import *

app = Flask(__name__)

NUM_OF_CLIENTS = 2  # At least 2 as the updateParam function work only then.


if os.path.exists('server.h5'):
    model = tf.keras.models.load_model('server.h5')
    print('Model loaded')
else:
    model = getNewModel()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    model.save('server.h5')
    print('Model Created!')

storage = []


@app.route('/', methods=['GET', 'POST'])
def server():
    global model
    if request.method == 'GET':
        return send_file('server.h5')

    else:
        global storage
        client = request.files['model'].read()
        open('from_client.h5', 'wb').write(client)

        client = tf.keras.models.load_model('client.h5')
        storage.append(client.get_weights())
        if len(storage) == NUM_OF_CLIENTS:  # All the clients have given their models
            res = updateParam(storage)
            model.set_weights(res)
            model.save('server.h5')
            storage = []
            # print('Model Updated')

        return "Model Received"


@app.route('/evaluate', methods=['GET'])
def evaluate():
    res = model.evaluate(X_test, Y_test, verbose=False)
    acc = res[1] * 100
    return 'Accuracy: ' + str(acc) + ' %'


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


if __name__ == '__main__':
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(use_reloader=True)
