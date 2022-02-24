"""

Load a prebuilt dataset
Build a neural network machine learning model that classifies images
Train this neural network
Evaluate the accuracy of the model

"""
import tensorflow as tf
print("Tensorflow version: ", tf.__version__)

#Loadin a dataset (MNIST dataset)
#Convert the sample data from integers to floating-point numbers
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
print("Loading done...")

# model : function with learnable parameters that maps an input to an output.
# training the model on data => optimal parameters

"""
parameter

A variable of a model that the machine learning system trains on its own. 
For example, weights are parameters whose values 
the machine learning system gradually learns 
through successive training iterations. 
"""

"""
Tensorflow provide two ways to create a machine learning model:
- Layers API (sequential/functional)
- Core API

"""

# sequential model : a linear stack of layers
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
print(predictions)
