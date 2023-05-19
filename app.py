import numpy as np
from flask import Flask, render_template, Response
import cv2
import threading
import time
from deepdream.image_processing import run_pyramid, processed_image_quque, img2frame

app = Flask(__name__)


def generate_frames():
    modified_image = None
    while True:
        if not processed_image_quque.empty():
            modified_image = img2frame(processed_image_quque.get())
        if modified_image is None:
            continue # wait for the first frame
        ret, buffer = cv2.imencode('.jpg', modified_image)
        frame = buffer.tobytes()
        # Yield the frame for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # Start the slow task in a separate thread
    slow_task_thread = threading.Thread(target=run_pyramid)
    slow_task_thread.daemon = True
    slow_task_thread.start()

    app.run(debug=True)
