import keras
import numpy as np

from time import time
import sys

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def get_python_pipeline_data(path='cat.jpg', batch_size=32, img_size=[480, 480], reps=100):

    start = time()

    for rep in range(reps):
        batch = np.zeros((batch_size, img_size[0], img_size[1], 3))
        for i in range(batch_size):
            img = load_img(path)
            batch[i, :, :, :] = img

    duration = time() - start
    im_sec = 1.0 / (duration / (reps * batch_size))
    print("%.2f" % im_sec)

def get_keras_pipeline_data(path='cat.jpg', batch_size=32, image_size=480, nclass=1000):

    datagen = ImageDataGenerator(
            rotation_range=0,
            width_shift_range=0.0,
            height_shift_range=0.0,
            shear_range=0.0,
            zoom_range=0.0,
            horizontal_flip=True,
            fill_mode='nearest')

    img = load_img(path)
    features = img_to_array(img)     
    features = features.reshape((1,) + features.shape)
    labels=np.random.randint(low=0, high=nclass-1, size=1)

    start = time() 
    ctr = 0
    for batch in datagen.flow(features, labels, batch_size=batch_size,
        save_to_dir=None, save_prefix=None, save_format=None):
        ctr += 1
        if ctr > 20:
            break 
    duration = time() - start
    im_sec = 1.0 / ((time() - start) / (ctr * batch_size))
    print("%.2f" % im_sec)

def main():

    print("Raw Python image pipeline:")
    get_python_pipeline_data()
    print("Keras image pipeline:")
    get_keras_pipeline_data()

if __name__ == '__main__':
    main()
