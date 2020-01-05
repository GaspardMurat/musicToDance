from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import ConvLSTM2D, MaxPooling3D, BatchNormalization
from keras import optimizers


def build_model(trainX, trainy, learning_rate, ):
    time_step = trainX.shape[1]
    height = trainX.shape[2]
    width = trainX.shape[3]
    depth = trainX.shape[4]
    output_shape = trainy.shape[1]

    main_input = Input(shape=(time_step, height, width, depth), name='main_input')

    # Encoder
    x = ConvLSTM2D(16, (32, 2), padding='same', return_sequences=True)(main_input)
    x = ConvLSTM2D(32, (32, 2), padding='same', return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(64, (32, 2), padding='same', return_sequences=True)(x)
    x = ConvLSTM2D(69, (32, 2), padding='same', return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D((1, 2, 2))(x)
    x = Flatten()(x)
    main_output = Dense(output_shape, name='main_output')(x)

    model = Model(inputs=main_input, outputs=main_output)

    optimizer = optimizers.adam(lr=learning_rate)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    return model


'''
 # Encoder
    x = Conv2D(8, (33, 3), activation='elu', padding='valid')(main_input)
    x = BatchNormalization()(x)
    x = Conv2D(16, (33, 3), padding='valid', activation='elu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (33, 2), padding='valid', activation='elu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (32, 2), padding='valid', activation='elu')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
'''