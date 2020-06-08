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

class Pix2Pix():

    def __init__(self, options):
        checkpoint_path = options['checkpoint']

        if checkpoint_path is not None: 
            self.model = tf.keras.models.load_model(checkpoint_path)
        else:
            self.model = tf.keras.models.load_model('generator_model_002_epochs_200.h5')


    # Generate an image based on a 256 by 256 input shape:
    def run_on_input(self, input_image):

        # TODO: image as array -> tensor
        # resize to 256x256
        
        input_image_processed = tf.image.convert_image_dtype(np.array(input_image), tf.float32)
        input_image_processed = tf.image.resize(input_image_processed, (256, 256), antialias=True)
        input_image_processed = tf.expand_dims(input_image_processed, axis=[0]) # (1, 256, 256, 3)
        prediction = self.model(input_image_processed, training=True)

        output_image = prediction[0] # -> (256, 256, 3)

        # TODO: Fix image styling issues - returned predictions look desaturated
        image_as_pil = tf.keras.preprocessing.image.array_to_img(output_image)    
        return image_as_pil