import tensorflow as tf
import numpy as np

def load_model():
    return tf.keras.models.load_model('model.h5')

def predict(image):
    model = load_model()
    image = np.expand_dims(image, axis=0) 
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)
    return predicted_class
