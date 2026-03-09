import tensorflow as tf
import numpy as np

NUM_ANOMALIES = 6
IMAGE_SIZE = 512

ANOMALY_NAMES = [
    "cloud_shadow",
    "double_plant",
    "planter_skip",
    "standing_water",
    "waterway",
    "weed_cluster",
]

#collect all images paths under data folder and store the path
dataset = tf.data.Dataset.list_files("data/*.jpg", shuffle = False)


def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels =3)
    #image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32)/255.0
    return image

