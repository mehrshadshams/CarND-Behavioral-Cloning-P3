import csv
import pandas as pd
import cv2
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Lambda, Dense, Conv2D, Flatten, MaxPool2D, Dropout
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


BATCH_SIZE = 32
EPOCHS = 10


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
                name = './data/IMG/' + batch_sample[0].split('/')[-1]
                image = cv2.imread(name)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
              	
                angle = batch_sample[1]

                if np.random.rand() > 0.5:
                    image = np.flipr(image)
                    angle *= -1

		image = image / 255. - 0.5
                images.append(image[60:-20,:,:])
                angles.append(batch_sample[1])

            X_train = np.array(images)
            y_train = np.array(angles)

            yield shuffle(X_train, y_train)


def create_model():
    inp = Input(shape=(80, 320, 3))
    x = Conv2D(24, (5, 5), activation='relu')(inp)
    x = MaxPool2D()(x)
    x = Conv2D(36, (5, 5), activation='relu')(x)
    x = MaxPool2D()(x)
    x = Conv2D(48, (3, 3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(50, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    output = Dense(1, activation='relu')(x)

    model = Model(inputs=inp, outputs=output)
    model.compile(optimizer='adam', loss='mse')

    return model


def main():
    print('Reading driving log...')

    samples = []
    df = pd.read_csv('data/driving_log.csv')

    with open('data/driving_log.csv', 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader, None)
        for row in csv_reader:
            center, left, right, steering, throttle, _break, speed = row
            steering = float(steering)
            samples.append([center, steering, throttle, _break, speed])
            samples.append([left, steering + 0.2, throttle, _break, speed])
            samples.append([right, steering - 0.2, throttle, _break, speed])

    train_samples, valid_samples = train_test_split(samples)

    train_generator = generator(train_samples, batch_size=BATCH_SIZE)
    valid_generator = generator(valid_samples, batch_size=BATCH_SIZE)

    model = create_model()

    checkpoint = ModelCheckpoint('weights.{epoch:02d}-{val_loss:.5f}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='max')

    model.fit_generator(generator=train_generator,
                        steps_per_epoch=len(train_samples) // BATCH_SIZE,
                        validation_data=valid_generator,
                        validation_steps=len(valid_samples) // BATCH_SIZE,
                        epochs=EPOCHS,
                        callbacks=[checkpoint])

    model.save('model.h5')


if __name__ == '__main__':
    main()