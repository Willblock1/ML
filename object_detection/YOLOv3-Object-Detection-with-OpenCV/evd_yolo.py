## EVD

"""
Interacts as client of EVD Engine (Enhanced Vehicle Detection)
Communicate with ML model and camera
listen for HME signal, then call EVD for binary reponse (yes the car is in true order state of no, the car is not ready to order)
Post back to API Broker to delivery signal to AIDT
"""
import socket
#import mock
import multiprocessing
import logging
from datetime import datetime
import ip
import requests
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import subprocess
import os
import cv2
import random

from yolo_utils import infer_image, show_image

logger = logging.getLogger(__name__)

EVD_HOST = ip.EVD_HOST
EVD_SOCKET_PORT = ip.EVD_SOCKET_PORT
API_HOST = 'localhost'
API_PORT = 5001
CAM_IP = '192.168.91.101'
CAM_USER = 'root'
CAM_PASSWORD = 'Slalom2019'

CV_INFER_SECONDS = 3 ## this is seconds
CONFIDENCE_THRESHOLD = 0.5 ## this is a percentage
RECEIVE_TIMEOUT_SECONDS = 2
KEEP_ALIVE_ITERATIONS = 5
RECONNECT_INTERVAL = 4
SESSION_ID = ''

class EVD(multiprocessing.Process):

    def __init__(self, incoming_queue_from_hme, outgoing_queue_to_consumers):
        multiprocessing.Process.__init__(self)
        self.out_queue = outgoing_queue_to_consumers
        self.in_queue = incoming_queue_from_hme
        self.name = 'AIDT_Broker_API_EVD_Client'

    def _establish_socket_connection(self):
        
        """
        Attempt to establish connection to HME.
        OR
        Attempt to establish connection to API Broker, depending on implimentation 
        """
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # s.connect((EVD_HOST, EVD_SOCKET_PORT))

            s.connect((API_HOST, API_PORT))

            #s.sendall(f'<SESSION Id="10" Cmd="OPEN" Pw="1124" Verbose="TRUE" /> {ETX}'.encode())
            #print(f'<SESSION Cmd="OPEN" Pw="1124" Verbose="TRUE" /> {ETX}'.encode())
            self._awaiting_reply = True
            s.settimeout(RECEIVE_TIMEOUT_SECONDS)
        except:
            logger.warning(f'Issue with attempt to establish socket connection at host {EVD_HOST} socket {EVD_SOCKET_PORT}', exc_info = True)
            s.close()
            return (False, s)
        else:
            logger.info('socket connection established')
            return (True, s)

    def _establish_video_stream(self):
        """
        Connect to IP Camera via network and open stream
        """
        print('accessing video stream...')
        try:
            #vs = VideoStream(f'http://{CAM_USER}:{CAM_PASSWORD}@{CAM_IP}/video.mjpg')
            vs = VideoStream(src = 0).start()
            return (True, vs)
        except:
            print('Failed to open camera stream, unable to infer')
            return (False, vs)

    def _load_model(self):
        """
        Load CV models in process such that EVD can infer on the fly
        """
        print('loading model...')
        net = ''
        args = ''
        layer_names = ''
        colors = ''
        labels = ''
        FLAGS = ''

        try:
            parser = argparse.ArgumentParser()

            parser.add_argument('-m', '--model-path',
                type=str,
                default='./yolov3-coco/',
                help='The directory where the model weights and \
                    configuration files are.')

            parser.add_argument('-w', '--weights',
                type=str,
                default='./yolov3-coco/yolov3.weights',
                help='Path to the file which contains the weights \
                        for YOLOv3.')

            parser.add_argument('-cfg', '--config',
                type=str,
                default='./yolov3-coco/yolov3.cfg',
                help='Path to the configuration file for the YOLOv3 model.')

            parser.add_argument('-i', '--image-path',
                type=str,
                help='The path to the image file')

            parser.add_argument('-v', '--video-path',
                type=str,
                help='The path to the video file')


            parser.add_argument('-vo', '--video-output-path',
                type=str,
                default='./output.avi',
                help='The path of the output video file')

            parser.add_argument('-l', '--labels',
                type=str,
                default='./yolov3-coco/coco-labels',
                help='Path to the file having the \
                            labels in a new-line seperated way.')

            parser.add_argument('-c', '--confidence',
                type=float,
                default=0.5,
                help='The model will reject boundaries which has a \
                        probabiity less than the confidence value. \
                        default: 0.5')

            parser.add_argument('-th', '--threshold',
                type=float,
                default=0.3,
                help='The threshold to use when applying the \
                        Non-Max Suppresion')

            parser.add_argument('--download-model',
                type=bool,
                default=False,
                help='Set to True, if the model weights and configurations \
                        are not present on your local machine.')

            parser.add_argument('-t', '--show-time',
                type=bool,
                default=False,
                help='Show the time taken to infer each image.')

            FLAGS, unparsed = parser.parse_known_args()

            # Download the YOLOv3 models if needed
            if FLAGS.download_model:
                subprocess.call(['./yolov3-coco/get_model.sh'])

            # Get the labels
            labels = open(FLAGS.labels).read().strip().split('\n')

            # Intializing colors to represent each label uniquely
            colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

            # Load the weights and configutation to form the pretrained YOLOv3 model
            net = cv2.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)

            # Get the output layer names of the model
            layer_names = net.getLayerNames()
            layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            return (True, net, args, layer_names, colors, labels, FLAGS)
        except KeyError as e:
            print(e, 'Model Failed to Load, unable to infer')
            return (False, net, args, layer_names, colors, labels, FLAGS)

    # def quit():


    def infer(self, vs, net, args, layer_names, colors, labels, FLAGS):
        """
        Pass video stream to model and infer car ready signals
        """
        count = 0
        fps = FPS().start()

        end = time.time() + CV_INFER_SECONDS

        output = {}

        while time.time() < end:

            frame = vs.read()
            height, width = frame.shape[:2]

            if count == 0:
                frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, \
                                    height, width, frame, colors, labels, FLAGS)
                count += 1
            else:
                frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, \
                                    height, width, frame, colors, labels, FLAGS, boxes, confidences, classids, idxs, infer=False)
                count = (count + 1) % 6

            cv2.imshow('webcam', frame)

            print(classids, idxs)

            #output[datetime.now()] = [classids[idxs], confidences[idxs]]

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
			# update the FPS counter
            fps.update()                

        # stop the timer and display FPS information
        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        # do a bit of cleanup
        print('clean environment: destroy windows')
        cv2.destroyAllWindows()

        output = {'time' : ['person', 100]}

        return (True, output)

    def evaluate(self, decision):
        """
        examine output of model and evaluate rules engine, output car-ready response if True
        """
        ## do mathy things and decide 
        ## mock in the meantime
        # print('EVALUATE FAKE MATH ON THIS DATA!')
        # print(decision)
        evaluated = True
        doubt = 0
        for label, confidence in decision.items():
            print('looking at: ', label, confidence)
            if float(confidence[1]) < 0.75:
                print('doubt it! Not too sure what this is')
                doubt = doubt + 1
            elif confidence[0] != 'person':
                print('doubt it! Thats not a person!')
                doubt = doubt + 1
        # evaluated = random.choice([True, False])
        print(doubt, len(decision))
        if len(decision) == 0:
            print('no value returned from model, default TRUE')
        elif doubt/len(decision) > CONFIDENCE_THRESHOLD:
            print(f'Looked at {len(decision)} inferences and that car is NOT READY!')
            evaluated = False
        return evaluated

    def run(self):
        """
        IF INTEGRATED API BROKER
        Continously check incoming queue for incoming data from HME
        Once found, request EVD response 

        IF CONSUMING EXTERNAL API BROKER
        listen for emmitted socket messages from API Broker

        """
        print('running EVD Engine...')

        (_ready, _sock) = self._establish_socket_connection()

        print(_ready)

        while not _ready:
            (_ready, _sock) = self._establish_socket_connection()

        (_open, _vs) = self._establish_video_stream()

        (_loaded, _net, _args, _layer_names, _colors, _labels, _FLAGS) = self._load_model()

        while True:

            if _ready:
                time.sleep(.5)
                try:

                    # if _open and _loaded:
                    #     (_bool, decision) = self.infer(_vs, _net, _args, _fps, CLASSES, COLORS)
                    # else:
                    #     print('could not open stream and load model, exit and try again')
                    #     time.sleep(1)
                    #     break
                    message = _sock.recv(2048)
                    if not message:
                        break

                    #message = self.in_queue.get(False)
                    print(f'found message from broker: {message}, gathering addtional evd info')
                    
                    if message and _open and _loaded:
                        (_bool, decision) = self.infer(_vs, _net, _args, _layer_names, _colors, _labels, _FLAGS)

                        if _bool:
                            _evaluated = self.evaluate(decision)

                            if _evaluated:
                                try:
                                    print(f'RETURNING INFORMED DECISON: {_evaluated} at {datetime.utcnow()}')
                                    requests.post('http://localhost:5000/broadcast', {'event' : 'car-ready', 'lane' : 1})
                                    print('POST CAR READY MESSAGE TO BROADCASTER')
                                except:
                                    print('unable to post message to broadcaster')
                except socket.timeout:
                    print('done listening to EVD')

            # try:
            #     while True:
            #         response = _sock.recv(1024)
            #         print(f'RETURNING INFORMED DECISON: {response} at {datetime.utcnow()}')
            #         requests.post('http://localhost:5000/broadcast', response)
            #         print('POST CAR READY MESSAGE TO BROADCASTER')
            # except socket.timeout:
            #     print('done listening to EVD')

evd = EVD(multiprocessing.JoinableQueue(), multiprocessing.JoinableQueue())

if __name__ == '__main__':
    evd.start()
########################################################

# import socket
# import sys
# import time
# import random
# import requests
# import io
# import ip
# import ctypes
# from PIL import Image
# from datetime import datetime

# # Create a TCP/IP socket
# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# # Bind the socket to the port
# server_address = ('localhost', 2626)
# print('starting up on %s port %s' % server_address)
# sock.bind(server_address)

# # Listen for incoming connections
# sock.listen(5)

# BROKER_API = 'localhost'
# CAM_HOST = ip.CAM_HOST
# CAM_PORT = ip.CAM_PORT
# IMG_REQUEST_TIMEOUT = 2

# ## LOAD MODELS
# ## ff = pi.load(/home/pi/models/ff.hdf5)
# ## cif = pi.load(home/pi/models/cif.hdf5)

# def get_image(timestamp):
#     try:
#         now = datetime.utcnow()
#         sent = 0
#         print(abs((timestamp - now).total_seconds()))
#         while abs((timestamp - now).total_seconds()) < IMG_REQUEST_TIMEOUT:
#             if sent == 0:
#                 print('CONNECTING TO IMAGE PIPELINE WITH', CAM_HOST, CAM_HOST)
#                 s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#                 s.connect((CAM_HOST, CAM_PORT))
#                 s.sendall(b'IMAGE REQUEST')
#             if sent == 1:
#                 response = s.recv(4096)
#                 print(f'RETURNING IMAGE DATA')
#                 if response:
#                     break
#             sent = 1
#         if abs((timestamp - now).total_seconds()) > IMG_REQUEST_TIMEOUT:
#             print('taking to long, exiting function')
#             response = b'time error'
#     except socket.timeout:
#         print('COULD NOT RETRIEVE IMAGE')
#         return
#     try:
#         s.close()
#     except:
#         print('socket either not successfully closed or was never opened')
#     return response

# while True:
#     # Wait for a connection
#     print('waiting for a connection')
#     connection, client_address = sock.accept()
#     with connection:
#         print('Connected by', client_address)
#         while True:
#             data = connection.recv(1024)
#             print(data, '<==== message sent to EVD')
#             if not data:
#                 break
#             # connection.sendall(data)
#             print('RECEIVED HME ALERT! CHECKING EVD')

#             img = get_image(datetime.utcnow())

#             print(type(img), '<<=== retrun from get_image()')

#             lane = 1 ## this will eventually parse the message for correct lane

#             ready = {}
#             # cif_sub_ready = None
#             # ff_sub_ready = None

#             ## PING MODEL HERE!
#             """

#             IF HTTP REQUEST 
#             ===============================
#             frame = requests.get(http:connection_to_feed_or_logs).data
#             frame = frame.some_parsing_method_for_ML_consumption
#             ff_pred = ff.predict(frame)

#             frames = requests.get(http:connection_to_feed_or_logs, args = 15_frames.zip).data
#             frames = frames.some_parsing_method_for_ML_consumption and successive framing
#             cif_pred = cif.predict(frames)

#             IF SOCKET COMMS
#             ===============================
#             socket.sendall(b'MEDIA REQUEST')
#             frame = socket.recv(1024) 
#             ^^ Need to be careful here as socket comm is byte based
#             which could cause image issues
                
#             frame = frame.split(some_kind_of_encoding)
#             """
#             ## PSUEDO CODE FOR IMAGE PARSING
#             # with open(img_path, 'rb') as i:
#             #     _bytes = bytearray(i.read())

#             # image = Image.open(io.BytesIO(_bytes))
#             # image.show()
#             """
#             ff_pred and cif_pred format:
#                 {
#                 ff_pred: [1 or 0, 0-100]
#                 ,cif_pred: [1 or 0, 0-100]
#                 }    
#             """
#             cif_pred = [random.randint(0,1), random.randint(80, 100)]
#             ff_pred = [random.randint(0,1), random.randint(80, 100)]

#             predictions = {'cif_pred' : cif_pred,
#                             'ff_pred': ff_pred}

#             print(cif_pred, ff_pred, '<<==== randomized model responses')
        
#             for name, prediction in predictions.items():
#                 sub_ready = None
#                 if prediction[0] == 0 and prediction[1] > 82:
#                     sub_ready = 0
#                 else: 
#                     pass
#                 if prediction[0] == 1 and prediction[1] > 82:
#                     sub_ready = 1
#                 else:
#                     pass
#                 ready[name] = sub_ready

#             decide = []

#             for name, prediction in ready.items():
#                 if prediction == 1:
#                     decide.append(1)

#             print(sum(decide), '<== decision points')

#             if sum(decide) == 2:
#                 ready = 1
#             else:
#                 ready = 0

#             if ready == 1:
#                 # message = '< MCAIDT SrcCmd="VEHICLE" Val="1" /> ETX'
#                 ctypes.windll.user32.MessageBoxW(0, 'Look, I found a car ready to order!', 'Car Ready!', 1)

#                 image = Image.open(io.BytesIO(img))
#                 image.show()
#                 print('CAR READY - RULES ENGINE REPORTS TRUE')
#                 if lane == 1:
#                     connection.sendall(b'< MCAIDT SrcCmd="VEHICLE" Val="31" /> ETX')
#                 if lane == 2:
#                     connection.sendall(b'< MCAIDT SrcCmd="VEHICLE" Val="32" /> ETX')
#                 print('SENT CAR READY REPONSE TO EVD CLIENT')
#             else:
#                 print('CAR NOT READY - RULES ENGINE REPORTS FALSE')
#                 pass
#             #time.sleep(random.uniform(2,4))



    