from keras.layers import  Dense, Dropout
from keras.models import Sequential
from utils.constants import NUM_CLASSES, DATA_SHAPE

keras_model = Sequential()
keras_model.add(Dense(1024, input_shape=(DATA_SHAPE,), activation="relu"))
keras_model.add(Dropout(0.2))
keras_model.add(Dense(512, activation="relu"))
keras_model.add(Dropout(0.2))
keras_model.add(Dense(256, activation="relu"))
keras_model.add(Dense(NUM_CLASSES, activation="softmax"))

keras_model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])