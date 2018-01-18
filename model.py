import csv
import cv2
import numpy as np
import pandas as pd
import argparse
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import utilities


BATCH_SIZE = 128
EPOCHS = 10
CORRECTION_FACTOR = 0.25 # try 0.1


def augment_brightness(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    random_brightness = .5 + np.random.uniform()
    image_hsv[:, :, 2] *= random_brightness
    image_hsv[:, :, 2] = np.clip(image_hsv[:, :, 2], 0, 255)
    return cv2.cvtColor(image_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def trans_image(image, steer, trans_range):
    # Translation
    rows, cols, _ = image.shape
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    steer_ang = steer + tr_x / trans_range * 2 * .2
    tr_y = 40 * np.random.uniform() - 40 / 2
    # tr_y = 0
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, Trans_M, (cols, rows))

    return image_tr, steer_ang


def add_shadow(image):
    h, w, _ = image.shape
    x1, y1 = w * np.random.rand(), 0
    x2, y2 = w * np.random.rand(), h

    xm, ym = np.mgrid[0:h, 0:w]
    mask = np.zeros_like(image[:, :, 0])

    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    cond = mask == np.random.randint(2)

    shadow_ratio = np.random.uniform(low=0.2, high=0.5)

    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    hls[:, :, 1][cond] = hls[:, :, 1][cond] * shadow_ratio

    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def extend_image(image, angle):

    # Only extend we a probability of 50%
    # if np.random.uniform() > 0.5:
    image, angle = trans_image(image, angle, 100)
    image = add_shadow(image)
    image = augment_brightness(image)

    # Don't flip image if we're moving in straight line (almost)
    if np.random.uniform() > 0.5:
    # if abs(angle) > 0.1:
        image = np.fliplr(image)
        angle *= -1

    return image, angle


def generator(data_path, X, y, batch_size=32, training=False):
    num_samples = len(X)
    while 1:
        X, y = shuffle(X, y)
        for offset in range(0, num_samples, batch_size):
            end = offset + batch_size
            batch_X, batch_y = X[offset:end], y[offset:end]

            images = []
            angles = []
            for row in range(len(batch_X)):
                files, angle = batch_X[row], batch_y[row]
                # if 1 == 1: # training:
                #     idx = np.random.randint(0, 3)
                # else:
                #     # only pass center image for validation
                #     idx = 1

                idx = np.random.randint(0, 3)
                filename = files[idx]

                if idx == 0:
                    angle += CORRECTION_FACTOR
                elif idx == 1:
                    angle -= CORRECTION_FACTOR

                name = data_path + '/IMG/' + filename.split('/')[-1]

                image = cv2.imread(name)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                if training:
                    image, angle = extend_image(image, angle)

                image = image[60:-20, :, :]
                image = cv2.resize(image, (64, 64))

                image = image / 255. - 0.5

                images.append(image)
                angles.append(angle)

            images = np.array(images)
            angles = np.array(angles)

            yield shuffle(images, angles)


def main(args):
    path = args.data + '/driving_log.csv'
    print('Reading driving log... ' + path)

    df = pd.read_csv(path)
    df.columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']

    X = df[['left', 'center', 'right']]
    y = df['steering']

    X_train, X_val, y_train, y_val = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.1, random_state=100)

    train_generator = generator(args.data, X_train, y_train, batch_size=BATCH_SIZE, training=True)
    valid_generator = generator(args.data, X_val, y_val, batch_size=BATCH_SIZE)

    model = utilities.create_model()

    checkpoint = ModelCheckpoint('weights.{epoch:02d}-{val_loss:.5f}.h5', monitor='val_loss', verbose=1,
                                 save_best_only=True)

    print('Training model for {0} epochs.'.format(args.epochs))

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=len(X_train) // BATCH_SIZE,
                                  validation_data=valid_generator,
                                  validation_steps=len(X_val) // BATCH_SIZE,
                                  epochs=args.epochs,
                                  callbacks=[checkpoint])

    model.save('model.h5')

    # utilities.plot_history(history.history, show=False)
    utilities.save_history(history.history)

    # print(len(train_samples))
    #
    # angles = []
    # for i in range(10):
    #     x = 0
    #     for X_train, y_train in train_generator:
    #         angles.extend(list(y_train))
    #         x += 1
    #         print(x)
    #         if x == len(train_samples) // BATCH_SIZE:
    #             break
    #
    # df = pd.DataFrame(angles)
    # df.hist()
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model arguments.')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Number of epochs')
    parser.add_argument('--data', default='./data',
                        help='Location of data folder')

    args = parser.parse_args()
    main(args)
