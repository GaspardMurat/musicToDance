import librosa
import codecs
import json
import numpy as np
import os
from keras.models import load_model
from sklearn import preprocessing
from dataLoader import dataLoader

sound_name = 'audio.mp3'
results_path = 'outputs/convlstm/Results_' + sound_name

model_path = 'outputs/convlstm/Checkpoints/'
model_name = 'model.ckpt.0100.hdf5'

path_to_data_test = 'dataset_master/DANCE_C_1'  # For testing

SPLIT = True
SPLIT_LEN = 3

repertory_exist = False


def soundTest(sond_name):
    sound_path = 'soundTest/' + sond_name

    #    Load the audio as a waveform `y`
    #    Store the sampling rate as `sr`
    y, sr = librosa.load(sound_path, sr=44100, dtype='float32')
    hop_length = 1764
    n_fft = 1024

    acoustic_features = []

    n = len(y)
    nb_frames = 0

    for x in range(0, n - hop_length, hop_length):
        slice = y[x:hop_length + x]
        stfft = librosa.feature.melspectrogram(y=slice, sr=sr, hop_length=256, n_fft=n_fft)
        acoustic_features.append(stfft)
        nb_frames += 1
    acoustic_features = np.concatenate(acoustic_features, axis=1)
    return acoustic_features, nb_frames


def split_sequence(sequence, n_steps):
    """

    :param sequence:
    :param n_steps:
    :return: a matrix where n_columns = n_step, and n_row = ( (len(sequence) - n_steps ) / stride ) + 1
                and stride = 1.
    """
    n_steps = n_steps - 1
    X = list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x = sequence[i:end_ix + 1]
        X.append(seq_x)
    return np.array(X)


def complete_sequence(sequence, n_steps):
    """
    After using split sequence, the sequence is left with (samples - (n_steps -1), n_steps, n_features).
    this will complete the sequence to (samples, n_steps, n_features) with a the  dat from thhe sequence.
    :param sequence:
    :param n_steps:
    :return: a sequence of size ((n_steps -1), n_steps, n_features) to add too the split sequence.
    """
    X = list()
    for i in range(n_steps - 1):
        step = list()
        n_i = n_steps - i
        for j in range(n_i):
            step.append(sequence[0])
        for j in range(n_i, n_steps):
            l = 1
            step.append(sequence[l])
            l += 1
        step = np.array(step)
        X.append(step)
    return np.array(X)


def normalize_audio(data):
    #normalizer = preprocessing.Normalizer().fit(data)
    #data = normalizer.transform(data)
    std_scale = preprocessing.StandardScaler().fit(data)
    data = std_scale.transform(data)
    return data


def reshape_acoustic_features(data, start_pos, end_pos):
    n_frames = end_pos - start_pos
    x = data.shape[0]
    y = int(data.shape[1] / n_frames)
    data = np.reshape(data, (n_frames, x, y))
    data = np.expand_dims(data, axis=3)
    return data


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


if __name__ == '__main__':
    model = load_model(model_path + model_name)
    if SPLIT:
        testX, testy, motions_max, motions_min, start_position, end_position = dataLoader(
            path_to_data=path_to_data_test,
            split=True,
            split_len=SPLIT_LEN,
            measures=True)
        acoustic_features, nb_frames = soundTest(sound_name)
        acoustic_features = normalize_audio(acoustic_features)
        acoustic_features = reshape_acoustic_features(acoustic_features, start_pos=0, end_pos=nb_frames)
        input = split_sequence(acoustic_features, SPLIT_LEN)
        add = complete_sequence(acoustic_features, SPLIT_LEN)

        input_sequence = np.concatenate((add, input))
        output = model.predict(input)
        one = np.ones(output.shape)
        output = (output + one) * (motions_max - motions_min) / 2 + motions_min
        print("input shape = ", input.shape)
        print("output shape = ", output.shape)
    else:
        testX, testy, motions_max, motions_min, start_position, end_position = dataLoader(
            path_to_data=path_to_data_test,
            split=False,
            split_len=3,
            measures=True)
        acoustic_features, nb_frames = soundTest(sound_name)
        acoustic_features = normalize_audio(acoustic_features)
        acoustic_features = reshape_acoustic_features(acoustic_features, start_pos=0, end_pos=nb_frames)

        input = acoustic_features
        output = model.predict(input)
        one = np.ones(output.shape)
        output = (output + one) * (motions_max - motions_min) / 2 + motions_min
        print("input shape = ", input.shape)
        print("output shape = ", output.shape)

    if repertory_exist:
        save(output, start_position, end_position, path=os.getcwd() + '/' + results_path)
    else:

        try:
            os.mkdir(os.getcwd() + '/' + results_path)
        except OSError:
            print("Creation of the directory %s failed" % os.getcwd() + '/' + results_path)
            quit()
        else:
            print("Successfully created the directory %s " % os.getcwd() + '/' + results_path)

        save(output, start_position, end_position, path=os.getcwd() + '/' + results_path)
