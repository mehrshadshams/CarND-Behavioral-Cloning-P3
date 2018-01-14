import csv
import pandas as pd
import cv2
import numpy as np
import argparse
from keras.models import Model, load_model
from keras.layers import Input, Lambda, Dense, Conv2D, Flatten, MaxPool2D, Dropout
# from keras.preprocessing import image as image_process
import transformations
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

BATCH_SIZE = 128
EPOCHS = 10


def extend_image(image, angle):
    # rotation
    image, r = transformations.random_rotation(image, 15, row_axis=0, col_axis=1, channel_axis=2)
    angle += (0.1 / 15) * r

    image, tx, ty = transformations.random_shift(image, 20, 20, row_axis=0, col_axis=1, channel_axis=2)
    angle += (tx * 0.005)

    return image, angle


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            end = offset + batch_size
            batch_samples = samples[offset:end]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data2/IMG/' + batch_sample[0].split('/')[-1]
                rev = batch_sample[-1]

                image = cv2.imread(name)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                angle = batch_sample[1]

                if rev:
#                if np.random.uniform() > 0.5:
                    image = np.fliplr(image)
                    angle *= -1

                image = image[60:-20, :, :]

                # if np.random.uniform() > 0.5:
                #    image, angle = extend_image(image, angle)

                image = image / 255. - 0.5
                #images.append(image[60:-20, :, :])
                images.append(image)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield shuffle(X_train, y_train)


def create_model():
    inp = Input(shape=(80, 320, 3))
    x = Conv2D(24, (5, 5), activation='elu', strides=(2, 2))(inp)
    # x = MaxPool2D()(x)
    x = Conv2D(36, (5, 5), activation='elu', strides=(2, 2))(x)
    # x = MaxPool2D()(x)
    x = Conv2D(48, (5, 5), activation='elu', strides=(2, 2))(x)

    x = Conv2D(64, (3, 3), activation='elu')(x)
    # x = MaxPool2D()(x)
    x = Conv2D(64, (3, 3), activation='elu')(x)
    x = Dropout(rate=0.5)(x)
    x = Flatten()(x)
    x = Dense(100, activation='elu')(x)
    x = Dense(50, activation='elu')(x)
    x = Dense(10, activation='elu')(x)
    output = Dense(1)(x)

    model = Model(inputs=inp, outputs=output)
    model.compile(optimizer='adam', loss='mse')

    return model


def main(args):
    print('Reading driving log...')

    samples = []
    df = pd.read_csv('data/driving_log.csv')

    with open('data2/driving_log.csv', 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader, None)
        for row in csv_reader:
            center, left, right, steering, throttle, _break, speed = row
            steering = float(steering)
            samples.append([center, steering, throttle, _break, speed, True])
            s2 = 0.25 # abs(steering * 0.5)

            samples.append([left, steering + s2, throttle, _break, speed, False])
            samples.append([right, steering - s2, throttle, _break, speed, False])
            samples.append([center, steering, throttle, _break, speed, True])
            samples.append([left, steering + s2, throttle, _break, speed, True])
            samples.append([right, steering - s2, throttle, _break, speed, True])

    train_samples, valid_samples = train_test_split(samples)

    train_generator = generator(train_samples, batch_size=BATCH_SIZE)
    valid_generator = generator(valid_samples, batch_size=BATCH_SIZE)

    model = create_model()

    checkpoint = ModelCheckpoint('weights.{epoch:02d}-{val_loss:.5f}.h5', monitor='val_loss', verbose=1,
                                 save_best_only=True)

    print('Training model for {0} epochs.'.format(args.epochs))

    model.fit_generator(generator=train_generator,
                        steps_per_epoch=len(train_samples) // BATCH_SIZE,
                        validation_data=valid_generator,
                        validation_steps=len(valid_samples) // BATCH_SIZE,
                        epochs=args.epochs,
                        callbacks=[checkpoint])

    model.save('model.h5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model arguments.')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Number of epochs')

    args = parser.parse_args()
    main(args)
