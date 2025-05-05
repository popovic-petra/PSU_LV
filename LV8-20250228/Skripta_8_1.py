from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# MNIST podatkovni skup
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train_s = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test_s = x_test.reshape(-1, 28, 28, 1) / 255.0

y_train_s = to_categorical(y_train, num_classes=10)
y_test_s = to_categorical(y_test, num_classes=10)

# TODO: strukturiraj konvolucijsku neuronsku mrezu

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),   # 32 filtra dim 3x3
    layers.MaxPooling2D((2, 2)), # sloj sazimanja
    layers.Conv2D(64, (3,3), activation='relu'),    # konv sloj sa 64 filtra dim 3x3
    layers.MaxPooling2D((2, 2)), # sloj sazimanja
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# TODO: definiraj karakteristike procesa ucenja pomocu .compile()

model.compile(
    optimizer='sgd',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# TODO: definiraj callbacks

my_callbacks = [
    callbacks.TensorBoard(log_dir='logs', update_freq=100),     # sprema logove o treningu
    callbacks.ModelCheckpoint(filepath='best_model.h5',         # spremi najbolji model 
                               monitor='val_accuracy', 
                               mode='max', 
                               save_best_only=True)
]

# TODO: provedi treniranje mreze pomocu .fit()

model.fit(  x_train_s,
            y_train_s,
            epochs = 50,
            batch_size = 64,
            callbacks = my_callbacks,
            validation_split = 0.1)

#TODO: Ucitaj najbolji model

best_model = keras.models.load_model('best_model.h5')

# TODO: Izracunajte tocnost mreze na skupu podataka za ucenje i skupu podataka za testiranje

train_accuracy = best_model.evaluate(x_train_s, y_train_s, verbose=0)
test_accuracy = best_model.evaluate(x_test_s, y_test_s, verbose=0)

print(f'Točnost na skupu za učenje: {train_accuracy[1]:.4f}')
print(f'Točnost na skupu za testiranje: {test_accuracy[1]:.4f}')

# TODO: Prikazite matricu zabune na skupu podataka za testiranje

y_pred = best_model.predict(x_test_s)
y_pred_classes = np.argmax(y_pred, axis=1)
conf_matrix = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(10, 7))
plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
plt.title("Matrica zabune")
plt.colorbar()
plt.xticks(np.arange(10))
plt.yticks(np.arange(10))
plt.xlabel("Predviđene klase")
plt.ylabel("Stvarne klase")
plt.show()