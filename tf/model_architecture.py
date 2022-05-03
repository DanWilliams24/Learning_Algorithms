from keras import layers
from keras.models import Model

# define a set of neurons creating the input layer -> of dim 1024 x 1
input = layers.Input(shape=(1024,), dtype='float32')
# defines the hidden layers of the model
middle = layers.Dense(units=512, activation='relu')(input)
# defines the output layers of the model


# time for HYPERNEAT
