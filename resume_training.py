#!/usr/bin/python
from dataLoader import dataLoader
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, TerminateOnNaN, LearningRateScheduler
import pickle

model_name = 'model.ckpt.0080.hdf5'
history_version = '0080'

path_to_model = 'outputs/convlstm/Checkpoints/'
path_to_history = 'outputs/convlstm/History/'
data_test = 'DANCE_C_1'

path_to_data_train = 'dataset_master/DANCE_C_1'  # For training
path_to_data_test = 'dataset_master/DANCE_C_1'  # For testing

split = True
split_len = 3

# Training parameters

EPOCHS = 20
learning_rate = 1e-04 * 0.9
validation_split = 0.0
batch_size = 32

trainX, trainy = dataLoader(path_to_data=path_to_data_train, split=split, split_len=split_len)
testX, testy, motions_max, motions_min, start_position, end_position = dataLoader(path_to_data=path_to_data_test,
                                                                                  split=split,
                                                                                  split_len=split_len,
                                                                                  measures=True)
model = load_model(path_to_model + model_name)
model_saver = ModelCheckpoint(filepath=path_to_model + model_name + ".{epoch:04d}.hdf5",
                              verbose=1,
                              save_best_only=False,
                              period=10)


def lr_scheduler(epoch, lr):
    return learning_rate


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

with open(path_to_history + 'trainHistoryDict' + history_version, 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
