import tensorflow as tf
from tensorflow import keras

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["QT_DEVICE_PIXEL_RATIO"] = "0"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
os.environ["QT_SCALE_FACTOR"] = "1"

model = keras.models.load_model('mnist_cnn')

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

def predict(image):
    image = image.reshape(1,28,28, 1)
    image = image / 255.0
    prediction = model.predict(image)
    return prediction