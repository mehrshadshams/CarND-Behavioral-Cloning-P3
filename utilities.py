import matplotlib.pyplot as plt
import pickle
from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense, Conv2D, Flatten, Lambda, Dropout


def plot_history(history, show=True):
    plt.plot(history['loss'], label='train')
    plt.plot(history['val_loss'], label='valid')
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.savefig('history.png')

    if show:
        plt.show()


def save_history(history):
    with open('history.pkl', 'wb') as f:
        pickle.dump(history, f)


def create_model():

    # model = Sequential()
    # model.add(Lambda(lambda x: x / 255. - 0.5, input_shape=(64, 64, 3)))
    # model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
    # model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
    # model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
    #
    # model.add(Conv2D(64, (3, 3), activation='elu'))
    # model.add(Conv2D(64, (3, 3), activation='elu'))
    #
    # model.add(Dropout(rate=0.5))
    # model.add(Flatten())
    # model.add(Dense(100, activation='elu'))
    # model.add(Dense(50, activation='elu'))
    # model.add(Dense(10, activation='elu'))
    # model.add(Dense(1))

    # inp = Input(shape=(80, 320, 3))
    inp = Input(shape=(64, 64, 3))
    x = Conv2D(24, (5, 5), activation='elu', strides=(2, 2))(inp)
    x = Conv2D(36, (5, 5), activation='elu', strides=(2, 2))(x)
    x = Conv2D(48, (5, 5), activation='elu', strides=(2, 2))(x)

    x = Conv2D(64, (3, 3), activation='elu')(x)
    x = Conv2D(64, (3, 3), activation='elu')(x)
    x = Dropout(rate=0.5)(x)
    x = Flatten()(x)
    x = Dense(100, activation='elu')(x)
    x = Dense(50, activation='elu')(x)
    x = Dense(10, activation='elu')(x)
    output = Dense(1)(x)

    model = Model(inputs=inp, outputs=output)

    model.compile(optimizer='adam', loss='mse')

    print(model.summary())

    return model
