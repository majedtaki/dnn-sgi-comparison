import argparse
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--channels-first', action='store_true')
    args = parser.parse_args()

    if args.channels_first:
        tf.keras.backend.set_image_data_format('channels_first')
    model: Model = ResNet50()
    model.save('./data/resnet50_tf/1')


if __name__ == '__main__':
    main()
