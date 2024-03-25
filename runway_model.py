# MIT License

# Copyright (c) 2019 Runway AI, Inc
# Copyright (c) 2020–2024 Nono Martínez Alonso, for the edited portions

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

# =========================================================================

# This example contains the minimum specifications and requirements
# to port a machine learning model to Runway.

# For more instructions on how to port a model to Runway, see the Runway Model
# SDK docs at https://sdk.runwayml.com

# RUNWAY
# www.runwayml.com
# hello@runwayml.com

# =========================================================================

# Import the Runway SDK. Please install it first with
# `pip install runway-python`.
import runway
from runway.data_types import number, text, image
from pix2pix_model import Pix2Pix
import argparse

# Setup the model, initialize weights, set the configs of the model, etc.
# Every model will have a different set of configurations and requirements.
# Check https://docs.runwayapp.ai/#/python-sdk to see a complete list of
# supported configs. The setup function should return the model ready to be
# used.

setup_options = {'checkpoint': runway.file(extension='.h5')}

@runway.setup(options=setup_options)
def setup(opts):
    model = Pix2Pix(opts)
    return model

# Every model needs to have at least one command. Every command allows to send
# inputs and process outputs. To see a complete list of supported inputs and
# outputs data types: https://sdk.runwayml.com/en/latest/data_types.html
@runway.command(name='generate',
                inputs={ 'input_image': image() },
                outputs={ 'output_image': image() },
                description='Generates a predicted image based on the given input image.')

def generate(model, args):
    # Generate an output image based on the input image, and return it
    output_image = model.run_on_input(args['input_image'])
    return {
        'output_image': output_image
    }

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Predict Pix2Pix for a given input image.')
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        default=None,
        help='Path to Pix2Pix model, e.g., edges2daisies.h5.',
    )

    opt = parser.parse_args()

    # Run the model server using the default network interface and ports,
    # displayed here for convenience.
    port=8000
    print(f"Running on port {port}..")
    runway.run(port=port, model_options={'checkpoint': opt.model})

    ## Now that the model is running, open a new terminal and give it a command to
    ## generate an image. It will respond with a base64 encoded URI
    # curl \
    #   -H "content-type: application/json" \
    #   -d '{ "caption": "red" }' \
    #   localhost:8000/generate