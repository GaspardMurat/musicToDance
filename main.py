#!/usr/bin/python
import codecs
import json
import os
import pickle
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, TerminateOnNaN, LearningRateScheduler

from dataLoader import dataLoader
from model import build_model

# Load data and build model
model_name = 'convlstm'
path_to_model = ''

data_test = 'DANCE_C_1'

path_to_data_train = 'dataset_master/DANCE_C_1'  # For training
path_to_data_test = 'dataset_master/DANCE_C_1'  # For testing

# Training model
'''
If mode = multiple, then the total number of epochs to train the model is EPOCHS*iteration.
Else, the number of epochs is EPOCHS, given in next.
'''

mode = 'callbacks'
SPLIT = True
SPLIT_LEN = 3

iteration = 2  # Used with multiple

# Training parameters
EPOCHS = 20
learning_rate = 0.0001
validation_split = 0
batch_size = 32


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label='Val Error')

    plt.ylim([0, 1])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'], label='Val Error')

    plt.ylim([0, 0.05])
    plt.legend()
    plt.show()


def save(output, start_pos, end_pos, path):
    nb_data = output.shape[0]
    output = np.reshape(output, (nb_data, 23, 3))
    output = output.tolist()
    skeletons = {"length": nb_data, "skeletons": output}
    with open(path + '/skeletons.json', "w") as write_file:
        json.dump(skeletons, codecs.open(path + '/skeletons.json', 'w', encoding='utf-8'),
                  separators=(',', ':'), sort_keys=True,
                  indent=4)

    start_pos = int(start_pos)
    end_pos = int(end_pos)
    config = {"start_position": start_pos, "end_position": end_pos}

    with open(path + '/config.json', "w") as write_file:
        json.dump(config, write_file)


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


trainX, trainy = dataLoader(path_to_data=path_to_data_train, split=SPLIT, split_len=SPLIT_LEN)
testX, testy, motions_max, motions_min, start_position, end_position = dataLoader(path_to_data=path_to_data_test,
                                                                                  split=SPLIT,
                                                                                  split_len=SPLIT_LEN,
                                                                                  measures=True)
input_shape = trainX.shape
output_shape = trainy.shape
print('input shape:', input_shape)
print('output shape:', output_shape)

os.chdir("outputs")


print("compiling model...")
path = os.getcwd() + "/" + model_name
try:
    os.mkdir(path)
except OSError:
    print("Creation of the directory %s failed" % path)
    quit()
else:
    print("Successfully created the directory %s " % path)
    os.mkdir(path + '/' + 'utils')
    copytree(src='../utils', dst=path + '/' + 'utils')

    if mode == 'simple':
        model = build_model(trainX, trainy, learning_rate=learning_rate)
        path = os.getcwd() + "/" + model_name
        os.chdir(path)
        # Model
        try:
            os.mkdir(os.getcwd() + "/Model")
        except OSError:
            print("Creation of the directory %s failed" % os.getcwd() + "/Model")
            quit()
        else:
            print("Successfully created the directory %s " % os.getcwd() + "/Model")
        # History
        try:
            os.mkdir(os.getcwd() + "/History")
        except OSError:
            print("Creation of the directory %s failed" % os.getcwd() + "/History")
            quit()
        else:
            print("Successfully created the directory %s " % os.getcwd() + "/History")
        # Results
        try:
            os.mkdir(os.getcwd() + "/Results_" + data_test)
        except OSError:
            print("Creation of the directory %s failed" % os.getcwd() + "/Results_" + data_test)
            quit()
        else:
            print("Successfully created the directory %s " % os.getcwd() + "/Results_" + data_test)

        history = model.fit(trainX,
                            trainy,
                            validation_split=validation_split,
                            epochs=EPOCHS,
                            batch_size=batch_size,
                            verbose=1)

        model.save(os.getcwd() + "/Model/" + model_name + '.h5')
        with open(os.getcwd() + "/History/" + 'trainHistoryDict', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        output = model.predict(testX)
        one = np.ones(output.shape)
        output = (output + one) * (motions_max - motions_min) / 2 + motions_min
        save(output, start_position, end_position, path=os.getcwd() + '/Results_' + data_test)
        quit()

    if mode == 'callbacks':
        model = build_model(trainX, trainy, learning_rate=learning_rate)
        path = os.getcwd() + "/" + model_name
        os.chdir(path)
        # Checkpoints
        try:
            os.mkdir(os.getcwd() + "/Checkpoints")
        except OSError:
            print("Creation of the directory %s failed" % os.getcwd() + "/Checkpoints")
            quit()
        else:
            print("Successfully created the directory %s " % os.getcwd() + "/Checkpoints")
        # History
        try:
            os.mkdir(os.getcwd() + "/History")
        except OSError:
            print("Creation of the directory %s failed" % os.getcwd() + "/History")
            quit()
        else:
            print("Successfully created the directory %s " % os.getcwd() + "/History")
        # Results
        try:
            os.mkdir(os.getcwd() + "/Results_" + data_test)
        except OSError:
            print("Creation of the directory %s failed" % os.getcwd() + "/Results_" + data_test)
            quit()
        else:
            print("Successfully created the directory %s " % os.getcwd() + "/Results_" + data_test)

        model_saver = ModelCheckpoint(filepath=os.getcwd() + "/Checkpoints/model.ckpt.{epoch:04d}.hdf5",
                                      verbose=1,
                                      save_best_only=False,
                                      period=10)


        def lr_scheduler(epoch, lr):
            decay_rate = 0.90
            decay_step = 20
            if epoch % decay_step == 0 and epoch:
                return lr * decay_rate
            return lr


        callbacks_list = [model_saver,
                          TerminateOnNaN(),
                          LearningRateScheduler(lr_scheduler, verbose=1)]  # Used with callbacks

        history = model.fit(trainX,
                            trainy,
                            validation_split=validation_split,
                            epochs=EPOCHS,
                            batch_size=batch_size,
                            verbose=1,
                            callbacks=callbacks_list,
                            shuffle=False)

        with open(os.getcwd() + "/History/" + 'trainHistoryDict', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        quit()
