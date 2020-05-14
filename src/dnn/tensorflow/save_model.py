import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50


def main():
    model: Model = ResNet50()
    model.save('./data/resnet50_tf/1')


if __name__ == '__main__':
    main()
