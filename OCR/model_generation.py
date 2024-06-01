import keras
import tensorflow as tf
import numpy as np
from keras.models import Sequential # type: ignore
from matplotlib import pyplot as plt
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten # type: ignore
from keras.utils import to_categorical # type: ignore
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics, svm
from keras import layers

path = 'data.npz'  
with np.load(path, allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)
encoded_mapping = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))
print("Encoded Mapping:", encoded_mapping)

num_classes = 14
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(64, kernel_size=(4, 4), activation="relu"), # am i doing the kernel size and pool size correctly?
        layers.MaxPooling2D(pool_size=(3, 3)),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(3, 3)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer="adam")

# train the model for 20 epochs
history = model.fit(x_train, y_train, batch_size=128, epochs=20, validation_split=0.2)
print("Model Performance on Test Data:")
result = model.evaluate(x=x_test, y=y_test)
print(model.summary())
#plot the accuracy performance of the model
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
#plot the loss performance of the model
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
model.save("model.keras")