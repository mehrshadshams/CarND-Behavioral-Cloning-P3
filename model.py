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
CORRECTION_FACTOR = 0.25


def augment_brightness(image):
    """
    Add random brightness
    """
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    random_brightness = .5 + np.random.uniform()
    image_hsv[:, :, 2] *= random_brightness
    image_hsv[:, :, 2] = np.clip(image_hsv[:, :, 2], 0, 255)
    return cv2.cvtColor(image_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def translate_image(image, steer):
    """
    Add random translation to image
    """
    max_shift = 50
    max_angle = 0.15

    rows, cols, _ = image.shape

    random_x = np.random.randint(-max_shift, max_shift + 1)
    steer += (random_x / max_shift) * max_angle

    M = np.float32([[1, 0, random_x], [0, 1, 0]])
    img = cv2.warpAffine(image, M, (cols, rows))
    return img, steer


def add_shadow(image):
    """
    Add random shadow
    """
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

    image, angle = translate_image(image, angle)
    image = add_shadow(image)
    image = augment_brightness(image)

    # flip the image by 50% chance
    if np.random.uniform() > 0.5:
        image = np.fliplr(image)
        angle *= -1

    return image, angle


def generator(data_path, X, y, batch_size=32, training=False):
    """
    This is the generator that loads, extends and returns the images and their corresponding steering wheel
    angle for a given batch
    """
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

                idx = np.random.randint(0, 3)
                filename = files[idx]

                if idx == 0:
                    angle += CORRECTION_FACTOR
                elif idx == 2:
                    angle -= CORRECTION_FACTOR

                name = data_path + '/IMG/' + filename.split('/')[-1]

                image = cv2.imread(name)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Only extend image in training
                if training:
                    image, angle = extend_image(image, angle)

                image = image[60:-20, :, :]
                image = cv2.resize(image, (64, 64))

                # clip the resulting angle between -1..1
                angle = np.clip(angle, -1.0, 1.0)

                images.append(image)
                angles.append(angle)

            images = np.array(images)
            angles = np.array(angles)

            yield shuffle(images, angles)


def main(args):
    path = args.data + '/driving_log.csv'
    print('Reading driving log... ' + path)

    # Load the csv
    df = pd.read_csv(path)
    df.columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']

    X = df[['left', 'center', 'right']]
    y = df['steering']

    # Shuffle the data before we split it
    X, y = shuffle(X, y)

    # Split data into train and validation set of size 10%
    X_train, X_val, y_train, y_val = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.1)

    # Create train generator
    train_generator = generator(args.data, X_train, y_train, batch_size=BATCH_SIZE, training=True)

    # Create validation generator
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

    utilities.save_history(history.history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model arguments.')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Number of epochs')
    parser.add_argument('--data', default='./data',
                        help='Location of data folder')

    args = parser.parse_args()
    main(args)
