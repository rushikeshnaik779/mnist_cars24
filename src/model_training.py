from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.models import Sequential
import warnings
warnings.filterwarnings('ignore')
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
import datetime

import mlflow.tensorflow
class MnistTrain:
    def __init__(self, X_train, y_train, X_test, y_test):

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def model_initializatioin(self):
        self.model = Sequential([
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



    def model_compilation(self):
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=CategoricalCrossentropy(),
            metrics=["acc"]

        )


    def model_training(self):
        with mlflow.start_run():
            self.model.fit(
                self.X_train,
                self.y_train,
                epochs = 1,
                batch_size=256,
                validation_data = (self.X_test, self.y_test)
            )
            test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test)
            mlflow.tensorflow.autolog()  # Logs the TensorFlow metrics automatically
            mlflow.log_metric("test_loss", test_loss)
            mlflow.log_metric("test_accuracy", test_acc)

            time_now = str(datetime.datetime.now())
            mlflow.tensorflow.save_model(self.model, f"model/{time_now}/")

            mlflow.register_model(f"./model/{time_now}", name="cnn_model")

    def model_save(self):
        import os
        if not os.path.exists("models"):
            os.mkdir("models")

        self.model.save("models/cnn_mymodel.h5")
        return self.model


    def run_training(self):

        self.model_initializatioin()
        self.model_compilation()
        self.model_training()
        self.model_save()

        return self.model


