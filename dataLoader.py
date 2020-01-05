import json
import librosa
import numpy as np
from sklearn import preprocessing


def load_motions_features(dance_path):
    """
    :param dance_path: a string that give the path to the folder for one dance
    :return: a dictionary motions_features = {frame : list( motions features ) }
    """
    config_path = dance_path + '/' + "config.json"
    skeletons_path = dance_path + '/' + 'skeletons.json'

    with open(config_path) as fin:
        config = json.load(fin)
    with open(skeletons_path, 'r') as fin:
        motion_features = np.array(json.load(fin)['skeletons'])

    start_pos = config['start_position']

    X = motion_features.shape[0]
    nb_features = motion_features.shape[1] * motion_features.shape[2]
    motions_features = np.reshape(motion_features, (X, nb_features))
    end_pos = motions_features.shape[0] + start_pos

    return motions_features, start_pos, end_pos


def load_acoustic_features(dance_path, start_pos, end_pos):
    sound_path = dance_path + '/' + 'audio.mp3'

    #    Load the audio as a waveform `y`
    #    Store the sampling rate as `sr`
    y, sr = librosa.load(sound_path, sr=44100, dtype='float32')
    slice_length = 1764

    first = int(start_pos * slice_length + slice_length / 2)
    last = int(end_pos * slice_length + slice_length / 2)

    if first - last > y.shape[0]:
        print('error')
        return 0, False
    else:
        acoustic_features = []
        y = y[first:last]
        n = len(y)
        for x in range(0, n - 1, slice_length):
            slice = y[x:slice_length + x]
            stft = librosa.feature.melspectrogram(y=slice, sr=sr, hop_length=256, n_mels=128)
            acoustic_features.append(stft)
        acoustic_features = np.concatenate(acoustic_features, axis=1)
        return acoustic_features, True


def normalize_skeletons(data):
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    one = np.ones(data.shape)
    normalize_data = (2 * (data - data_min) / (data_max - data_min)) - one
    return normalize_data, data_max, data_min


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


def dataLoader(path_to_data='dataset_master', split=False, split_len=3, measures=False):
    if split:

        # load the input sequence
        # Motion features shape is (samples, 69), a skeleton for each frame made of 23 points in 3 dimension
        motions_features, START_POS, END_POS = load_motions_features(path_to_data)
        motions_features, MOTION_MIN, MOTION_MAX = normalize_skeletons(motions_features)
        # Acoustic features shape is (samples, 127, 8, 1), a sample correspond to a spectrogram for a frame. the
        # fourth dim is only for the convolutionnal block, which require a shape of (samples,  height, width, depth).
        acoustic_features, bool = load_acoustic_features(path_to_data, START_POS, END_POS)
        acoustic_features = normalize_audio(acoustic_features)
        acoustic_features = reshape_acoustic_features(acoustic_features, START_POS, END_POS)
        # Acoustic features a transformed a second time to include a timeSteps dim.
        # The acoustic features shape will be (samples, timeSteps,  height, width, depth).
        trainX = split_sequence(acoustic_features, split_len)
        add = complete_sequence(acoustic_features, split_len)

        trainX = np.concatenate((add, trainX))
        trainy = motions_features

    else:

        # load the input sequence
        # Motion features shape is (samples, 69), a skeleton for each frame made of 23 points in 3 dimension
        motions_features, START_POS, END_POS = load_motions_features(path_to_data)
        motions_features, MOTION_MIN, MOTION_MAX = normalize_skeletons(motions_features)
        # Acoustic features shape is (samples, 127, 8, 1), a sample correspond to a spectrogram for a frame. the
        # fourth dim is only for the convolutionnal block, which require a shape of (samples,  height, width, depth).
        acoustic_features, bool = load_acoustic_features(path_to_data, START_POS, END_POS)
        acoustic_features = normalize_audio(acoustic_features)
        #acoustic_features = reshape_acoustic_features(acoustic_features, START_POS, END_POS)

        trainX = acoustic_features
        trainy = motions_features

    if measures:
        return trainX, trainy, MOTION_MAX, MOTION_MIN, START_POS, END_POS

    else:
        return trainX, trainy


if __name__ == '__main__':
    path = 'dataset_master/DANCE_C_1'
    trainX, trainy, motions_max, motions_min, start_position, end_position = dataLoader(path, split=False, split_len=3,
                                                                                        measures=True)
    print("trainX shape: ", trainX.shape)
    print("trainy shape: ", trainy.shape)
    print("everything work")
    print(np.mean(trainX))
    print(trainX.min())
    print(trainX.max())


