import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense, Conv2D, Flatten, Lambda, Dropout
from keras.optimizers import Adam


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
    """
    This method creates the Keras model described in NVIDIA paper
    http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    :return:
    """

    model = Sequential()
    model.add(Lambda(lambda x: x / 255. - 0.5, input_shape=(64, 64, 3)))
    model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))

    model.add(Dropout(rate=0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    adam = Adam(lr=1e-3, decay=0.95)

    model.compile(optimizer=adam, loss='mse')

    print(model.summary())

    return model


def create_activation_model(model):
    """
    Model of all of the activations of the convolution layers
    :param model:
    :return:
    """
    layer_outputs = [layer.output for layer in model.layers[1:5]]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    return activation_model


def create_activation_map(activation_model, image_tensor):
    """
    This method create a map of all of the activations for a given image
    :param activation_model:
    :param image_tensor:
    :return:
    """
    activations = activation_model.predict(image_tensor)

    images_per_row = 8
    images = []

    max_width = -1
    for layer_activation in activations:
        n_features = layer_activation.shape[-1]

        # Layer activation has size (1, size, size, n_features)
        size = layer_activation.shape[1]

        n_rows = int(math.ceil(n_features / images_per_row))
        width = images_per_row * size

        max_width = max(max_width, width)

        display_grid = np.zeros((size * n_rows, width))

        for f in range(n_features):
            row = f // images_per_row
            col = f % images_per_row

            channel_image = layer_activation[0, :, :, f]
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')

            display_grid[row * size:(row + 1) * size, col * size:(col + 1) * size] = channel_image

        images.append(display_grid)

    temp = []
    for img in images:
        img2 = np.zeros((img.shape[0], max_width))
        img2[:, 0:img.shape[1]] = img
        temp.append(img2)

    final = np.concatenate(temp, axis=0)

    return np.dstack([final, np.zeros_like(final), np.zeros_like(final)]).astype('uint8')
