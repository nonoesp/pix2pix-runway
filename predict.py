from pix2pix_model import Pix2Pix
from PIL import Image
import tensorflow as tf
import numpy as np
import argparse

def predict(model_path, input_path, output_path):
    
    print("Loading Pix2Pix checkpoint..")
    
    model = Pix2Pix({'checkpoint': model_path}).model

    input_image = Image.open(input_path)

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

    output_image.save(output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predict Pix2Pix for a given input image.')
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        default='input.png',
        help='Input path.',
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='output.png',
        help='Output path.',
    )
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        default=None,
        help='Path to Pix2Pix model, e.g., edges2daisies.h5.',
    )

    opt = parser.parse_args()

    if opt.model and opt.input and opt.output:
        predict(opt.model, opt.input, opt.output)