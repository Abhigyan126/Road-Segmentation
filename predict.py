#grey
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load the saved model
generator_ = tf.keras.models.load_model('/Users/abhigyan/Downloads/GAN_Sat_image_grey_300.h5')

def load_and_predict(image_path):
    combined_image = tf.cast(img_to_array(load_img(image_path)), tf.float32)

    image = combined_image

    image = tf.image.rgb_to_grayscale(tf.image.resize(image,(256,256)))/255

    predicted = generator_.predict(tf.expand_dims(image, axis=0))[0]

    plt.figure(figsize=(10, 8))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Satellite Image")
    plt.axis('off')


    plt.subplot(1, 3, 3)
    plt.imshow(predicted)
    plt.title("Predicted Image")
    plt.axis('off')

    plt.show()

# Example usage
image_path = "/Users/abhigyan/Downloads/download copy.png"
load_and_predict(image_path)