from keras.utils import to_categorical

class MnistFEngineering:

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test


    def normalize(self, X):
        return X/255

    def categorical_conversion(self, y):
        return to_categorical(y)


    def run_pipeline_training(self):
        self.X_train = self.normalize(self.X_train)
        self.y_train = self.categorical_conversion(self.y_train)
        self.X_test = self.normalize(self.X_test)
        self.y_test = self.categorical_conversion(self.y_test)
        return self.X_train, self.y_train, self.X_test, self.y_test


    def run_pipeline_inference(self):
        self.X = self.normalize()






