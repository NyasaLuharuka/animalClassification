import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

class_names = ['cats', 'dogs']

model = load_model('animalModel.h5')

img = tf.keras.utils.load_img('dogTest.jpeg', target_size=(180, 180))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print(class_names[np.argmax(score)])