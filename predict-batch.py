from pix2pix_model import Pix2Pix
from PIL import Image
import tensorflow as tf
import numpy as np
import glob
from pathlib import PurePosixPath
import os

print("Loading Pix2Pix checkpoint..")
model = Pix2Pix({'checkpoint': '210121_092658_edges2sunflowers_tf-2.4.0@last.h5'}).model
# model = Pix2Pix({'checkpoint': None}).model

save_dir = '/Users/nono/Desktop/output'

for path in glob.glob('/Users/nono/Downloads/tmp/*'):
    save_path = os.path.join(save_dir, PurePosixPath(path).name)
    input_image = Image.open(path)

    # ↓
    # Pre-processing input
    img_in = tf.image.convert_image_dtype(np.array(input_image), tf.float32)
    img_in = tf.image.resize(img_in, (256, 256), antialias=True)
    img_in = tf.expand_dims(img_in, axis=[0]) # (1, 256, 256, 3)
    # ↓
    # Predict
    print("Predicting..")
    prediction = model(img_in, training=True)
    # ↓
    # Remap output from (-1.0, +1.0) to (0, 255)
    img_out = prediction[0] # -> (256, 256, 3)
    img_out = (img_out * 0.5 + 0.5) * 255.0
    img_out = tf.cast(img_out, tf.uint8)

    # Construct a PIL image to display in Runway
    output_image = Image.fromarray(np.array(img_out))
    output_image.save(save_path)