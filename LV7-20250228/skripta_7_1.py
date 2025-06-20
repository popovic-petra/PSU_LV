import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Dense
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


# MNIST podatkovni skup
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# TODO: prikazi nekoliko slika iz train skupa

for i in range(5, 10):
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"broj: {y_train[i]}")
    plt.axis('off')
    plt.show()

# Skaliranje vrijednosti piksela na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# Slike 28x28 piksela se predstavljaju vektorom od 784 elementa
x_train_s = x_train_s.reshape(60000, 784)
x_test_s = x_test_s.reshape(10000, 784)

# Kodiraj labele (0, 1, ... 9) one hot encoding-om
y_train_s = keras.utils.to_categorical(y_train, 10)
y_test_s = keras.utils.to_categorical(y_test, 10)


# TODO: kreiraj mrezu pomocu keras.Sequential(); prikazi njenu strukturu pomocu .summary()

model = Sequential()
model.add(keras.Input(shape=(784,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# TODO: definiraj karakteristike procesa ucenja pomocu .compile()

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# TODO: provedi treniranje mreze pomocu .fit()

fit = model.fit(x_train_s, y_train_s, epochs=5, batch_size=32)

# TODO: Izracunajte tocnost mreze na skupu podataka za ucenje i skupu podataka za testiranje

train_loss, train_accuracy = model.evaluate(x_train_s, y_train_s, verbose=0)
test_loss, test_accuracy = model.evaluate(x_test_s, y_test_s, verbose=0)
print(f"Točnost na trening skupu: {train_accuracy:.2f}")
print(f"Točnost na testnom skupu: {test_accuracy:.2f}")

# TODO: Prikazite matricu zabune na skupu podataka za testiranje

y_test_pred = model.predict(x_test_s)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)

conf_matrix = ConfusionMatrixDisplay(y_test, y_test_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=range(10))
disp.plot()
plt.title("Matrica zabune na skupu za testiranje")
plt.show()

# TODO: Prikazi nekoliko primjera iz testnog skupa podataka koje je izgrađena mreza pogresno klasificirala

netocno = np.where(y_test != y_test_pred_classes)[0]
for i in range(5):
    index = netocno[i]
    plt.imshow(x_test[index], cmap='gray')
    plt.title(f"Stvarno: {y_test[index]}, Predviđeno: {y_test_pred_classes[index]}")
    plt.axis('off')
    plt.show()
