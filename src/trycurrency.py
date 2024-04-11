import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('efficient_net_b0.h5')

# Function to preprocess the input image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to predict currency denomination
def predict_currency(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    classes = ['10', '20', '50', '100', '200', '500', '2000', 'Not a currency']
    return classes[class_index]

# Input image file path
img_path = 'path_to_your_image.jpg'

# Predict currency denomination
currency = predict_currency(img_path)
print("Predicted currency denomination:", currency)
