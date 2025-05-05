import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# 1. Učitaj spremljeni model
model = tf.keras.models.load_model("best_model.h5")

# 2. Učitaj i pripremi sliku
img_path = "00031_00002_00021.png"
img = image.load_img(img_path, target_size=(48, 48))  # resize da odgovara modelu
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # oblik (1, 48, 48, 3)
img_array = img_array / 255.0

predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)

print(f"Predikcija: {predicted_class}")
plt.imshow(img)
plt.title(f"Predikcija klase: {predicted_class}")
plt.show()

# tocna klasa je 31