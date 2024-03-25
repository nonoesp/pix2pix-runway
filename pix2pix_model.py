# MIT License

# Copyright (c) 2019 Runway AI, Inc

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import random
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import base64
from datetime import datetime

class Pix2Pix():

    def __init__(self, options):
        checkpoint_path = options['checkpoint']
        print(checkpoint_path)
        if checkpoint_path is not None: 
            self.model = tf.keras.models.load_model(checkpoint_path)


    # Generate an image based on a 256 by 256 input shape:
    def run_on_input(self, input_image):

        # Get current date and set save folder
        # now = datetime.today().strftime('%y%m%d_%H%M%S%f') 
        
        # ↓
        # Pre-processing input
        img_in = tf.image.convert_image_dtype(np.array(input_image), tf.float32)
        img_in = tf.image.resize(img_in, (256, 256), antialias=True)
        img_in = tf.expand_dims(img_in, axis=[0]) # (1, 256, 256, 3)
        # ↓
        # Predict
        prediction = self.model(img_in, training=True)
        # ↓
        # Remap output from (-1.0, +1.0) to (0, 255)
        img_out = prediction[0] # -> (256, 256, 3)
        img_out = (img_out * 0.5 + 0.5) * 255.0
        img_out = tf.cast(img_out, tf.uint8)

        # Construct a PIL image to display in Runway
        output_image = Image.fromarray(np.array(img_out))

        # Save input and out images to disk
        # input_image.save(f'/Users/nono/Desktop/pix/{now}-in.jpg')        
        # output_image.save(f'/Users/nono/Desktop/pix/{now}-out.jpg')

        return output_image