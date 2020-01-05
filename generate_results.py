import codecs
import json
import os

import numpy as np
from keras.models import load_model

from dataLoader import dataLoader

model_path = '/outputs/convlstm/Checkpoints/'
model_name = 'model.ckpt.0020.hdf5'

data_test = 'DANCE_C_1_20'
results_path = '/outputs/convlstm/Results_' + data_test
path_to_data_test = '/dataset_master/DANCE_C_1'  # For testing

split = True
split_len = 3

repertory_exist = False


def save(output, start_position, end_position, path):
    nb_data = output.shape[0]
    output = np.reshape(output, (nb_data, 23, 3))
    output = output.tolist()
    skeletons = {"length": nb_data, "skeletons": output}
    with open(path + '/skeletons.json', "w") as write_file:
        json.dump(skeletons, codecs.open(path + '/skeletons.json', 'w', encoding='utf-8'),
                  separators=(',', ':'), sort_keys=True,
                  indent=4)

    start_position = int(start_position)
    end_position = int(end_position)
    config = {"start_position": start_position, "end_position": end_position}

    with open(path + '/config.json', "w") as write_file:
        json.dump(config, write_file)


path_to_data_test = os.getcwd() + path_to_data_test
path_to_model = os.getcwd() + model_path + model_name
model = load_model(path_to_model)
testX, testy, motions_max, motions_min, start_position, end_position = dataLoader(path_to_data=path_to_data_test,
                                                                                  split=split,
                                                                                  split_len=split_len,
                                                                                  measures=True)
output = model.predict(testX)
one = np.ones(output.shape)
output = (output + one) * (motions_max - motions_min) / 2 + motions_min
if repertory_exist:
    save(output, start_position, end_position, path=os.getcwd() + results_path)
else:

    try:
        os.mkdir(os.getcwd() + results_path)
    except OSError:
        print("Creation of the directory %s failed" % os.getcwd() + results_path)
        quit()
    else:
        print("Successfully created the directory %s " % os.getcwd() + results_path)

    save(output, start_position, end_position, path=os.getcwd() + results_path)
