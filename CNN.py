import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from scipy.misc import imresize

#matplotlib inline

import os
from os import listdir
from os.path import isfile, join
import shutil
import stat
import collections
from collections import defaultdict

from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets

import h5py
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.layers import Input
import tools.image_gen_extended as T
from keras.models import load_model
import multiprocessing as mp

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau
from keras.optimizers import SGD
from keras.regularizers import l2
import keras.backend as K
import math

import pdb


def load_images(root, min_side=299):
    all_imgs = []
    all_classes = []
    resize_count = 0
    invalid_count = 0

    for i, subdir in enumerate(listdir(root)):
        imgs = listdir(join(root, subdir))
        #pdb.set_trace()
        class_ix = class_to_ix[subdir]
        print(i, class_ix, subdir)
        #pdb.set_t race()
        for img_name in imgs:
            img_arr = img.imread(join(root, subdir, img_name))
            img_arr_rs = img_arr
            try:
                w, h, _ = img_arr.shape
                if w < min_side:
                    wpercent = (min_side/float(w))
                    hsize = int((float(h)*float(wpercent)))
                    #print('new dims:', min_side, hsize)
                    img_arr_rs = imresize(img_arr, (min_side, hsize))
                    resize_count += 1
                elif h < min_side:
                    hpercent = (min_side/float(h))
                    wsize = int((float(w)*float(hpercent)))
                    #print('new dims:', wsize, min_side)
                    img_arr_rs = imresize(img_arr, (wsize, min_side))
                    resize_count += 1
                all_imgs.append(img_arr_rs)
                all_classes.append(class_ix)
            except:
                print('Skipping bad image: ', subdir, img_name)
                invalid_count += 1
    print(len(all_imgs), 'images loaded')
    print(resize_count, 'images resized')
    print(invalid_count, 'images skipped')
    return np.array(all_imgs), np.array(all_classes)


def schedule(epoch):
    if epoch < 15:
        return .01
    elif epoch < 28:
        return .002
    else:
        return .0004


def center_crop(x, center_crop_size):
    centerw, centerh = x.shape[0]//2, x.shape[1]//2
    halfw, halfh = center_crop_size[0]//2, center_crop_size[1]//2
    return x[centerw-halfw:centerw+halfw+1,centerh-halfh:centerh+halfh+1, :]


def predict_10_crop(img, ix, top_n, plot, preprocess, debug):
    flipped_X = np.fliplr(img)
    crops = [
        img[:299,:299, :], # Upper Left
        img[:299, img.shape[1]-299:, :], # Upper Right
        img[img.shape[0]-299:, :299, :], # Lower Left
        img[img.shape[0]-299:, img.shape[1]-299:, :], # Lower Right
        center_crop(img, (299, 299)),
        
        flipped_X[:299,:299, :],
        flipped_X[:299, flipped_X.shape[1]-299:, :],
        flipped_X[flipped_X.shape[0]-299:, :299, :],
        flipped_X[flipped_X.shape[0]-299:, flipped_X.shape[1]-299:, :],
        center_crop(flipped_X, (299, 299))
    ]
    if preprocess:
        crops = [preprocess_input(x.astype('float32')) for x in crops]

    if plot:
        fig, ax = plt.subplots(2, 5, figsize=(10, 4))
        ax[0][0].imshow(crops[0])
        ax[0][1].imshow(crops[1])
        ax[0][2].imshow(crops[2])
        ax[0][3].imshow(crops[3])
        ax[0][4].imshow(crops[4])
        ax[1][0].imshow(crops[5])
        ax[1][1].imshow(crops[6])
        ax[1][2].imshow(crops[7])
        ax[1][3].imshow(crops[8])
        ax[1][4].imshow(crops[9])
    
    y_pred = model.predict(np.array(crops))
    preds = np.argmax(y_pred, axis=1)
    top_n_preds= np.argpartition(y_pred, -top_n)[:,-top_n:]
    if debug:
        print('Top-1 Predicted:', preds)
        print('Top-5 Predicted:', top_n_preds)
        print('True Label:', y_test[ix])
    return preds, top_n_preds

if __name__ == "__main__":
    num_processes = 3
    pool = mp.Pool(processes=num_processes)

    class_to_ix = {}
    ix_to_class = {}
    with open('food/classes.txt', 'r') as txt:
        classes = [l.strip() for l in txt.readlines()]
        class_to_ix = dict(zip(classes, range(len(classes))))
        ix_to_class = dict(zip(range(len(classes)), classes))
        class_to_ix = {v: k for k, v in ix_to_class.items()}

    pdb.set_trace()

    X_image, y_label = load_images('food/image', min_side=299)
    X_train = X_image
    y_train = y_label
    X_test = X_image
    y_test = y_label

    # from keras.utils.np_utils import to_categorical

    n_classes = len(classes)
    y_train_cat = to_categorical(y_train, num_classes=n_classes)
    y_test_cat = to_categorical(y_test, num_classes=n_classes)

    pdb.set_trace()

    # this is the augmentation configuration we will use for training
    train_datagen = T.ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        zoom_range=[.8, 1],
        channel_shift_range=30,
        fill_mode='reflect')
    train_datagen.config['random_crop_size'] = (299, 299)
    train_datagen.set_pipeline([T.random_transform, T.random_crop, T.preprocess_input])
    train_generator = train_datagen.flow(X_train, y_train_cat, batch_size=64, seed=11, pool=pool)

    test_datagen = T.ImageDataGenerator()
    test_datagen.config['random_crop_size'] = (299, 299)
    test_datagen.set_pipeline([T.random_transform, T.random_crop, T.preprocess_input])
    test_generator = test_datagen.flow(X_test, y_test_cat, batch_size=64, seed=11, pool=pool)

    pdb.set_trace()

    K.clear_session()

    base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(299, 299, 3)))
    x = base_model.output
    x = AveragePooling2D(pool_size=(8, 8))(x)
    x = Dropout(.4)(x)
    x = Flatten()(x)
    predictions = Dense(n_classes, kernel_initializer='glorot_uniform', kernel_regularizer=l2(.0005),
                        activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    opt = SGD(lr=.01, momentum=.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath='model4.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)
    csv_logger = CSVLogger('model4.log')

    lr_scheduler = LearningRateScheduler(schedule)

    pdb.set_trace()

    model.fit_generator(train_generator,
                        validation_data=test_generator,
                        validation_steps=X_test.shape[0],
                        steps_per_epoch=X_train.shape[0],
                        epochs=10,
                        verbose=2,
                        callbacks=[lr_scheduler, csv_logger, checkpointer])

    model = load_model(filepath='./model4.10-0.02.hdf5')




    ix = 10
    predict_10_crop(img = X_test[ix], ix = ix, top_n=5, plot=True, preprocess=True, debug=True)


