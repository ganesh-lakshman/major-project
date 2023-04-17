import socketio
import eventlet

import numpy as np
from flask import Flask
# app = Flask(__name__)
# @app.route('/home')
# def greeting():
#     return "welcome"
# if __name__ == '__main__':
#     app.run(port=3000)
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2

sio = socketio.Server()

app = Flask(__name__)  # '__main__'
speed_limit = 5


def img_preprocess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img


# @sio.on('telemetry')
# def telemetry(sid, data):
#     try:
#         print("Telemetry received:")
#         print(data)
#         # process telemetry data
#     except Exception as e:
#         print("Error processing telemetry:", e)

@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    # steering_angle = float(model.predict(image, batch_size=1))
    throttle = 1.0 - speed/speed_limit
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    # print("now connected to", sid, environ)
    # send_control(0, 1)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })


if __name__ == '__main__':
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)