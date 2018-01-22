from datetime import datetime
import os
import argparse
from utilities import create_activation_map, create_activation_model
import glob
import h5py
import shutil
from keras.models import load_model
import cv2

OUTPUT = 'output'

activation_model = None


def main(args):
    for filename in glob.glob(os.path.join(args.image_folder, '*.jpg')):
        print('Processing {}'.format(filename))

        image_array = cv2.imread(filename)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        image_array = image_array[60:-20, :, :]

        image_array = cv2.resize(image_array, (64, 64))

        image_array = image_array / 255. - 0.5

        image_tensor = image_array[None, :, :, :]

        activation_img = create_activation_map(activation_model, image_tensor)

        timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
        image_filename = os.path.join(OUTPUT, timestamp)

        cv2.imwrite('{}.jpg'.format(image_filename), activation_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    f = h5py.File(args.model, mode='r')

    if not os.path.exists(OUTPUT):
        os.mkdir(OUTPUT)
    else:
        shutil.rmtree(OUTPUT)
        os.mkdir(OUTPUT)

    model = load_model(args.model)
    activation_model = create_activation_model(model)

    main(args)
