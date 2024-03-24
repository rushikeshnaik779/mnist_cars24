import tensorflow as tf

class MnistInfer:
    def __init__(self, X_test):
        self.X_test = X_test

    def inference_data_prep(self, X_test):

        # normalize the data.
        self.X_test = X_test/255
        self.X_test = tf.reshape(self.X_test, [1, 28, 28]) # reshaping to a required format

    def load_model(self, model_path):
        self.loaded_model = tf.keras.models.load_model(model_path)



    def predictions(self):
        prediction = self.loaded_model(self.X_test)
        prediction = tf.argmax(prediction[0]).numpy()
        return prediction


