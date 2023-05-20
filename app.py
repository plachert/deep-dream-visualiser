from flask import Flask, render_template, request, redirect, url_for
import base64
import cv2
import numpy as np
import threading
from deepdream.image_processing import run_pyramid


app = Flask(__name__)

# Define the options for the selection list
options = ['option1', 'option2', 'option3']

# Define the selected option variable
selected_option = 'option1'
image_generation_event = threading.Event()

# List of image arrays
image_arrays = [
    np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8),  # Example array, replace with your own
    np.array([[0, 255, 255], [255, 0, 255], [255, 255, 0]], dtype=np.uint8),  # Example array, replace with your own
    # Add more image arrays here
]


def convert_array_to_base64(array):
    # Convert the array to an image
    image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    # Encode the image to base64
    retval, buffer = cv2.imencode('.png', image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    return encoded_image

@app.route('/')
def index():
    global image_arrays
    current_index = int(request.args.get('index', 0))
    current_array = image_arrays[current_index]
    current_image = convert_array_to_base64(current_array)

    total_images = len(image_arrays)
    is_last_index = current_index == total_images - 1

    return render_template('index.html', current_image=current_image, current_index=current_index, total_images=total_images, is_last_index=is_last_index, options=options, selected_option=selected_option)


@app.route('/next')
def next_image():
    global image_arrays
    current_index = int(request.args.get('index', 0))
    next_index = current_index + 1

    if next_index >= len(image_arrays):
        next_index = len(image_arrays) - 1

    next_array = image_arrays[next_index]
    next_image = convert_array_to_base64(next_array)

    total_images = len(image_arrays)
    is_last_index = next_index == total_images - 1

    return render_template('index.html', current_image=next_image, current_index=next_index, total_images=total_images, is_last_index=is_last_index)

@app.route('/previous')
def previous_image():
    global image_arrays
    current_index = int(request.args.get('index', 0))
    previous_index = current_index - 1

    if previous_index < 0:
        previous_index = 0

    previous_array = image_arrays[previous_index]
    previous_image = convert_array_to_base64(previous_array)

    total_images = len(image_arrays)
    is_last_index = previous_index == total_images - 1

    return render_template('index.html', current_image=previous_image, current_index=previous_index, total_images=total_images, is_last_index=is_last_index)


@app.route('/generate', methods=['POST'])
def generate():
    global image_arrays, image_generation_event
    # selected_option = request.form['option']
    # Start the generate_images function in a separate thread
    image_generation_event.clear()
    thread = threading.Thread(target=generate_images)
    thread.start()
    image_generation_event.wait()
    return redirect(url_for('index'))


def generate_images(selected_option=selected_option):
    global image_arrays, image_generation_event
    processed_images = run_pyramid()
    def fix_processed(image):
        def deprocess(image):
            img = image.copy()
            img[0, :, :] *= 0.229
            img[1, :, :] *= 0.224
            img[2, :, :] *= 0.225
            img[0, :, :] += 0.485
            img[1, :, :] += 0.456
            img[2, :, :] += 0.406
            return img
        deprocessed_image = deprocess(image)
        rescaled_image = (deprocessed_image * 255).astype(np.uint8)
        transposed_image = np.transpose(rescaled_image, (1, 2, 0))
        return transposed_image
    new_image_arrays = [fix_processed(processed_image) for processed_image in processed_images]
    image_arrays = new_image_arrays
    image_generation_event.set()
    


# Start the image generation thread
if __name__ == '__main__':
    app.run(debug=True)

