import glob
import cv2
import os

import pca as pca
import tensorflow as tf
import h5py as h5py
import numpy as np
import scipy
import keras

from keras.layers import ZeroPadding2D, Conv2D,MaxPooling2D,BatchNormalization,Input,Flatten,Dense,Dropout,AveragePooling2D,Activation,Add,add
from keras import optimizers
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Layer
from keras.preprocessing.image import ImageDataGenerator
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy.misc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from defs import *

#os.environ["CUDA_VISIBLE_DEVICES"]="1"


def Unit(x, filters, pool=False):
  res = x
  if pool:
    x = MaxPooling2D(pool_size=(2, 2))(x)
    res = Conv2D(filters=filters, kernel_size=[1, 1], strides=(2, 2), padding="same",

                 )(res)
  out = BatchNormalization()(x)
  out = Activation("relu")(out)
  out = Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same",)(out)

  out = BatchNormalization()(out)
  out = Activation("relu")(out)
  out = Conv2D(filters=filters, kernel_size=[1, 1], strides=[1, 1], padding="same",)(out)

  out = add([res, out])

  return out


class ODEBlock(Layer):

    def __init__(self, filters, kernel_size, **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        super(ODEBlock, self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv2d_w1 = self.add_weight("conv2d_w1", self.kernel_size + (self.filters + 1, self.filters), initializer='glorot_uniform')
        self.conv2d_w2 = self.add_weight("conv2d_w2", self.kernel_size + (self.filters + 1, self.filters), initializer='glorot_uniform')
        self.conv2d_b1 = self.add_weight("conv2d_b1", (self.filters,), initializer='zero')
        self.conv2d_b2 = self.add_weight("conv2d_b2", (self.filters,), initializer='zero')
        self.built = True
        super(ODEBlock, self).build(input_shape)

    def call(self, x):
        t = K.constant([0, 1], dtype="float32")
        return tf.contrib.integrate.odeint(self.ode_func, x, t, rtol=1e-3, atol=1e-3)[1]

    def compute_output_shape(self, input_shape):
        return input_shape

    def ode_func(self, x, t):
        y = self.concat_t(x, t)
        y = K.conv2d(y, self.conv2d_w1, padding="same")
        y = K.bias_add(y, self.conv2d_b1)
        y = K.relu(y)

        y = self.concat_t(y, t)
        y = K.conv2d(y, self.conv2d_w2, padding="same")
        y = K.bias_add(y, self.conv2d_b2)
        y = K.relu(y)

        return y

    def concat_t(self, x, t):
        new_shape = tf.concat(
            [
                tf.shape(x)[:-1],
                tf.constant([1],dtype="int32",shape=(1,))
            ], axis=0)

        t = tf.ones(shape=new_shape) * tf.reshape(t, (1, 1, 1, 1))
        return tf.concat([x, t], axis=-1)

def Model_NN(input_shape):
  images = Input(input_shape)

  net = Conv2D(16, kernel_size=[7, 7], strides=(1, 1), padding="same", activation='relu')(images)
  net = Unit(net, 8, pool=True)
  net = Unit(net, 8)
  net = ODEBlock(8, (3, 3))(net)
  net = AveragePooling2D(pool_size=(4, 4))(net)
  net = Flatten()(net)
  net = Dense(units=7, activation="softmax")(net)
  model = Model(inputs=images, outputs=net)

  return model



#Build HDF5
'''
path_data_train = 'dataset_costruito_me/train'
X_train,Y_train = get_data(path_data_train)

path_data_test = 'dataset_costruito_me/test'
X_test,Y_test = get_data(path_data_test)

f = h5py.File('X_train_my.hdf5', 'w')
X_train = f.create_dataset("train_X", data=X_train)

f = h5py.File('Y_train_my.hdf5', 'w')
Y_train = f.create_dataset("train_Y", data=Y_train)

f = h5py.File('X_test_my.hdf5', 'w')
X_test = f.create_dataset("test_X", data=X_test)
f = h5py.File('Y_test_my.hdf5', 'w')
Y_test = f.create_dataset("test_Y", data=Y_test)'''

#READING da HDF5
X_train_f = h5py.File('X_train_my.hdf5', 'r')
X_train = X_train_f.get('train_X').value

Y_train_f = h5py.File('Y_train_my.hdf5', 'r')
y_train = Y_train_f.get('train_Y').value

X_train_f = h5py.File('X_test_my.hdf5', 'r')
X_test = X_train_f.get('test_X').value

Y_train_f = h5py.File('Y_test_my.hdf5', 'r')
y_test = Y_train_f.get('test_Y').value


num_classes = len(np.unique(y_train))
print("Number of classes: ",num_classes)



# normalize the data
'''X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

X_train = X_train - X_train.mean()
X_test = X_test - X_test.mean()

train_x = X_train / X_train.std(axis=0)
test_x = X_test / X_test.std(axis=0)'''

'''datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

datagen.fit(train_x,augment=True)'''

y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

w,h,c = 128,128,3
epochs = 30
bs = 32
sgd = optimizers.SGD(lr=0.0001, momentum=0.9, decay=0.0, nesterov=False)
input_shape = (w,h,c)
model = Model_NN(input_shape)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=epochs,batch_size=bs,validation_data=(X_test,y_test),shuffle=True)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.show()
plt.savefig('plot_accuracy.png')

plt.figure(2)
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('plot_loss.png')

accuracy = model.evaluate(x=X_test, y=y_test, verbose=0)
target_names = ['ArianaGrande', 'BillGates','DonaldTrump','EmmaStone','SelenaGomez','TaylorSwift','LuigiRusso']



y_pred = model.predict(X_test)
Y_true = np.argmax(y_test, axis=1)
Y_pred_classes = np.argmax(y_pred, axis=1)

roc_each_classes(y_test,y_pred,num_classes)


print('\n', classification_report(np.where(y_test > 0)[1], np.argmax(y_pred, axis=1),
                                  target_names=target_names))

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

plot_confusion_matrix(confusion_mtx, classes=target_names)

print('Test loss:', accuracy[0])
print('Test accuracy:', accuracy[1])

model.save_weights("detection_person_model.h5")