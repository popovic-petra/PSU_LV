import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Rescaling
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from tensorflow.keras.preprocessing import image_dataset_from_directory 
 

from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import os
import datetime


model = Sequential()
model.add(Rescaling(1./255, input_shape=(48, 48, 3)))

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
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']
)

# UCITAVANJE SLIKA
# path_train = "P:/3 - bezobrazovanje/0 - FERIT/2. godina 2024-2025/2. semestar/Strojno ucenje/gtsrb/Train"
# path_test = "P:/3 - bezobrazovanje/0 - FERIT/2. godina 2024-2025/2. semestar/Strojno ucenje/gtsrb/Test"

train_dir = "C:\\Users\\student\\Desktop\\PP_psu\\gtsrb\\Train"
test_dir = "C:\\Users\\student\\Desktop\\PP_psu\\gtsrb\\Test" 

train_ds = image_dataset_from_directory( 
    directory=train_dir, 
    labels='inferred', 
    label_mode='categorical', 
    batch_size=32, 
    subset="training", 
    seed=123, 
    validation_split=0.2, 
    image_size=(48, 48)) 
 
validation_ds = image_dataset_from_directory( 
    directory=train_dir, 
    labels='inferred', 
    label_mode='categorical', 
    batch_size=32, 
    subset="validation", 
    seed=123, 
    validation_split=0.2, 
    image_size=(48, 48)) 
 
test_ds = image_dataset_from_directory( 
    directory=test_dir, 
    labels='inferred', 
    label_mode='categorical', 
    batch_size=32, 
    image_size=(48, 48)
)

# Callbackovi
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)
log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

# TRENITRANJE modela
history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=10,
     callbacks=[checkpoint, tensorboard]
)

# TESTIRANJE - evaluacija na testnom skupu
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.2f}")

# Ekstrakcija stvarnih labela i predikcija
y_true = []
y_pred = []

best_model = tf.keras.models.load_model('best_model.h5')

for images, labels in test_ds:
    preds = best_model.predict(images)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(labels.numpy(), axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Matrica zabune
cm = confusion_matrix(y_true, y_pred)
print("Matrica zabune:\n", cm)

# Točnost klasifikacije
acc = accuracy_score(y_true, y_pred)
print(f"Točnost klasifikacije na testnom skupu: {acc * 100:.2f}%")
# Točnost klasifikacije na testnom skupu: 97.10%