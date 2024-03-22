import pandas as pd
import numpy as np
import tensorflow as tf
import ssl
import requests

# workaround as tensorflow was not allowing to read the data. due to python package
# this file needs to be added in the folder of keras data_utils ----> TODO
requests.packages.urllib3.disable_warnings()
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    print("error")
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context



def get_data():
    """
        function will read the data from the mnist dataset present in tensorflow keras package
        :return:
            train: 
                X_train, 
                y_train
            test: 
                X_test, 
                y_test 
    """

    (X_train,y_train),(X_test,y_test)=tf.keras.datasets.mnist.load_data()

    return (X_train,y_train), (X_test, y_test)






