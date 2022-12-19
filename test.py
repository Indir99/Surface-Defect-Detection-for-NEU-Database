import os
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop
from keras_preprocessing.image import ImageDataGenerator
from sklearn.datasets import load_files
from keras.utils import np_utils
from keras_preprocessing.image import array_to_img, img_to_array, load_img
from keras.layers import BatchNormalization

path = os.getcwd()
dir_list = os.listdir(path)
print("Files and directories in '", path, "' :")
# print the list
print(dir_list)

train_dir = path+'/NEU Metal Surface Defects Data/train'
val_dir = path+'/NEU Metal Surface Defects Data/valid'
test_dir = path+'/NEU Metal Surface Defects Data/test'
print("Path: ", os.listdir(path+"/NEU Metal Surface Defects Data"))
print("Train: ", os.listdir(path+"/NEU Metal Surface Defects Data/train"))
print("Test: ", os.listdir(path+"/NEU Metal Surface Defects Data/test"))
print("Validation: ", os.listdir(path+"/NEU Metal Surface Defects Data/valid"))

print("Inclusion Defect")
print("Training Images:", len(os.listdir(train_dir+'/'+'Inclusion')))
print("Testing Images:", len(os.listdir(test_dir+'/'+'Inclusion')))
print("Validation Images:", len(os.listdir(val_dir+'/'+'Inclusion')))

# Rescaling all Images by 1./255
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Training images are put in batches of 10
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(200, 200),
    batch_size=10,
    class_mode='categorical')

# Validation images are put in batches of 10
validation_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=(200, 200),
    batch_size=10,
    class_mode='categorical')


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.98):
            print("\nReached 98% accuracy so cancelling training!")
            self.model.stop_training = True


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                           input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(6, activation='softmax')
])
model.summary()


model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

tf.keras.utils.plot_model(
    model,
    to_file='cnn_architecture.png',
    show_shapes=True)


callbacks = myCallback()
history = model.fit(train_generator,
                    batch_size=32,
                    epochs=10,
                    validation_data=validation_generator,
                    callbacks=[callbacks],
                    verbose=1, shuffle=True)

sns.set_style("whitegrid")
plt.subplot(211)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

sns.set_style("whitegrid")
plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Loading file names & their respective target labels into numpy array


def load_dataset(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = np.array(data['target'])
    target_labels = np.array(data['target_names'])
    return files, targets, target_labels


x_test, y_test, target_labels = load_dataset(test_dir)
no_of_classes = len(np.unique(y_test))
no_of_classes

y_test = np_utils.to_categorical(y_test, no_of_classes)


def convert_image_to_array(files):
    images_as_array = []
    for file in files:
        # Convert to Numpy Array
        images_as_array.append(img_to_array(load_img(file)))
    return images_as_array


x_test = np.array(convert_image_to_array(x_test))
print('Test set shape : ', x_test.shape)

x_test = x_test.astype('float32')/255

loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
# loss_v, accuracy_v = model.evaluate(x_validate, y_validate, verbose=1)
# print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))
print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))


# Plotting Random Sample of test images, their predicted labels, and ground truth
y_pred = model.predict(x_test)
fig = plt.figure(figsize=(10, 10))
for i, idx in enumerate(np.random.choice(x_test.shape[0], size=16, replace=False)):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_test[idx]))
    pred_idx = np.argmax(y_pred[idx])
    true_idx = np.argmax(y_test[idx])
    ax.set_title("{} ({})".format(target_labels[pred_idx], target_labels[true_idx]),
                 color=("green" if pred_idx == true_idx else "red"))
fig.show()
plt.show()
