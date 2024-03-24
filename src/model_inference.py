import tensorflow as tf

class MnistInfer:
    def __init__(self, X_test):
        self.X_test = X_test

    def load_model(self):
        self.loaded_model = tf.keras.models.load_model('my_model.h5')


    def predictions(self):
        prediction = self.loaded_model(self.X_test)


    def run_pipeline(self):
        self.load_model()
        self.predictions()

