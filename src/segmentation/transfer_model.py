from keras.applications import MobileNetV2
from keras.layers import Conv1D, Reshape, GlobalAveragePooling2D, Dense
from keras.models import Sequential
from utils.constants import NUM_CLASSES

#base_model = ResNet50V2(weights='imagenet', input_shape=input_shape, include_top = False)
base_model = MobileNetV2(weights='imagenet', input_shape=(62, 62, 3), include_top = False)
base_model.trainable = False

# needs to be transformed to image since the model expects images
# 88 * 44 equals 3872 - the length of spectra
# inputs = keras.Input(shape=(3872, 1, 1))
transfer_model = Sequential()
transfer_model.add(Conv1D(filters=3, kernel_size=29, activation='relu', input_shape=(3872,1)))
transfer_model.add(Reshape((62, 62, 3)))
transfer_model.add(base_model)
transfer_model.add(GlobalAveragePooling2D())
transfer_model.add(Dense(NUM_CLASSES, activation='softmax'))

transfer_model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])