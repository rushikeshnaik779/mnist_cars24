import matplotlib.pyplot as plt
# extract data
from src import data_extraction
from src import model_training
from src.feature_engineering import MnistFEngineering
from src import model_inference

#
(X_train, y_train), (X_test, y_test) = data_extraction.get_data()
fengineering = MnistFEngineering(X_train, y_train, X_test, y_test)
X_train, y_train, X_test, y_test = fengineering.run_pipeline_training()


train = model_training.MnistTrain(
    X_train = X_train,
    y_train = y_train,
    X_test = X_test,
    y_test = y_test
)
model = train.run_training()



(X_train, y_train), (X_test, y_test) = data_extraction.get_data()

# print(X_test[0].shape)
import pickle
import os
import tensorflow as tf
# ... (same code as before to select image and label)

# Save image and label as a dictionary
data = X_test[6]
folder_name = "./data"
import os
if not os.path.exists(folder_name):
    os.mkdir(folder_name)
# Save the dictionary as a pickle file
with open(os.path.join(folder_name, "test_single_instance.pkl"), "wb") as f:
    pickle.dump(data, f)

print(f"Saved image and label to {folder_name} in pickle format")
print(y_test[6])

with open(os.path.join(folder_name, "test_single_instance.pkl"), "rb") as f:
    data = pickle.load(f)

print((data/255).reshape)

image = tf.reshape(data, [1, 28, 28])

new_model = tf.keras.models.load_model('./models/cnn_mymodel.h5')

predictions = new_model.predict(image/255)

print("Prediction by a model : ", tf.argmax(predictions[0]).numpy())

plt.imshow(X_test[6])

plt.savefig(f"digit {y_train[2]}")
plt.show()

