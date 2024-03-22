import numpy as np

from src import data_extraction as de
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = de.get_data()

# printing the details for the project
print(X_train.shape)
print(X_test.shape)
print(X_train[0])

# plt.imshow(X_train[0])
#
# plt.savefig(f"digit {y_train[0]}")
# plt.show()


# normalize
X_train = X_train/255
X_test = X_test/255

# plt.imshow(X_train[0])
#
# plt.savefig(f"Normalized digit {y_train[0]}")
# plt.show()

# print checking the classes we have
from keras.utils import to_categorical
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.models import Sequential
cnn = Sequential([
Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu', input_shape=(28, 28, 1)),
MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
Dropout(0.2),
Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'),
MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
Dropout(0.2),
Flatten(),
Dense(units=128, activation='relu'),
Dense(units=10, activation='softmax'),
])



import warnings
warnings.filterwarnings('ignore')
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy

cnn.compile(optimizer=Adam(learning_rate=0.001),
            loss=CategoricalCrossentropy(),
            metrics=["acc"])

history = cnn.fit(X_train, y_train, epochs=1, batch_size=256, validation_data=(X_test, y_test))
# Save the model
cnn.save('my_model.h5')

import matplotlib.pyplot as plt
plt.figure(figsize=(5, 5))
plt.plot(history.history['loss'], color='blue', label='loss');
plt.plot(history.history['val_loss'], color='red', label='val_loss');
plt.legend();
plt.title('Loss vs Validation Loss');
plt.tight_layout()
# plt.show()

import tensorflow as tf
# Load the model
new_model = tf.keras.models.load_model('my_model.h5')


# prediction
new_model = tf.keras.models.load_model('my_model.h5')

predictions = new_model.predict(X_test[0, :])

print(predictions)
