import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import os
import datetime

# UCITAVANJE SLIKA

train_dir = "P:/3 - bezobrazovanje/0 - FERIT/2. godina 2024-2025/2. semestar/Strojno ucenje/gtsrb/Train"
test_dir = "P:/3 - bezobrazovanje/0 - FERIT/2. godina 2024-2025/2. semestar/Strojno ucenje/gtsrb/Test"

img_height, img_width = 48, 48
batch_size = 64

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

model = Sequential()

# # 1. konvolucijski sloj
# model.add(Conv2D(
#     filters=32,                 # x = 32 filtera
#     kernel_size=(3, 3),         # 3x3 dimenzije filtera
#     strides=(1, 1),             # stride 1
#     padding='same',             # da izlaz ima iste dimenzije kao ulaz
#     activation='relu',          # relu aktivacija
#     input_shape=(48, 48, 3)     # ulazna slika: visina x sirina x broj kanala (RGB)
# ))

# # 2. konvolucijski sloj  
# model.add(Conv2D(
#     filters=32,                 # x = 32
#     kernel_size=(3, 3),         # 3x3 filter
#     strides=(1, 1),             # korak 1
#     padding='valid',            # BEZ paddinga -> smanjenje dimenzije
#     activation='relu'
# ))

# # 3. MaxPooling sloj – smanjenje dimenzija
# model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

# # 4. Dropout sloj – regularizacija -> crveni sloj
# model.add(Dropout(0.2))

# # 5. Ravnanje (Flatten) -> zuti sloj
# model.add(Flatten())

##--------------------------------------------------------------------------------##
#  objedinjujemo gore napisane formule u funkciju za x filtera [32, 64, 128] 

# ARHITEKTURA MODELA

filter_size = [32, 64, 128]

# dodaj prvi sloj koji ima input shape
model.add(Conv2D(filters=filter_size[0],
                 kernel_size=(3, 3),
                 strides=1,
                 padding='same',
                 activation='relu',
                 input_shape=(48, 48, 3)))

# petlja ponavlja 2. - 5. blok sa razlicitim filterima
for x in filter_size:
    model.add(Conv2D(filters=x,
                     kernel_size=(3, 3),
                     strides=1,
                     padding='valid',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.2))
    
model.add(Flatten())

# Potpuno povezani slojevi
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(43, activation='softmax'))  # 43 klasa prometnih znakova

# Kompajliranje podatka
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy']
)

# Callbackovi
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)
log_dir = TensorBoard(log_dir='logs', update_freq=100)
tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

# TRENITRANJE modela
model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    callbacks=[checkpoint, tensorboard]
)

# TESTIRANJE - evaluacija na testnom skupu
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# Ucitavamo najbolji model
best_model = tf.keras.models.load_model('best_model.h5')

# Predikcije
predictions = best_model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# Matrica zabune
cm = confusion_matrix(y_true, y_pred)
print("Matrica zabune:\n", cm)

# Točnost klasifikacije
acc = accuracy_score(y_true, y_pred)
print(f"Točnost klasifikacije na testnom skupu: {acc * 100:.2f}%")