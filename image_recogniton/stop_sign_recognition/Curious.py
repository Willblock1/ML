
# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils.video import VideoStream
from threading import Thread
import gopigo3
import numpy as np
import imutils
import time
import cv2
import os


# create and instance of the gopigo
GPG = gopigo3.GoPiGo3()

# define the paths to the Not stop Keras deep learning model
MODEL_PATH = "/home/pi/Desktop/CuriousGeorge/new_modelpy2.1.h5"

# initialize the total number of frames that *consecutively* contain
# stop along with threshold required to trigger the stop
TOTAL_CONSEC = 0
TOTAL_THRESH = 5
 
# initialize is the stop has been triggered
STOP = False

# load the model
print("[INFO] loading model...")
model = load_model(MODEL_PATH)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# prepare the image to be classified by our deep learning network
	image = cv2.resize(frame, (28, 28))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
 
	# classify the input image and initialize the label and
	# probability of the prediction
	(notStop, stop) = model.predict(image)[0]
	label = "Not Stop"
	proba = notStop
	GPG.set_motor_power(GPG.MOTOR_LEFT + GPG.MOTOR_RIGHT, 50)

	# check to see if stop was detected using our convolutional
	# neural network
	if stop > notStop:
		# update the label and prediction probability
		label = "Stop"
		proba = stop
 
		# increment the total number of consecutive frames that
		# contain stop
		TOTAL_CONSEC += 1

		# check to see if we should raise the stop
		if not STOP and TOTAL_CONSEC >= TOTAL_THRESH:
			# indicate that stop has been found
			STOP = True
			GPG.set_motor_power(GPG.MOTOR_LEFT + GPG.MOTOR_RIGHT, 0)
                        GPG.set_led(GPG.LED_BLINKER_LEFT, 100)
                        GPG.set_led(GPG.LED_BLINKER_RIGHT, 100)
			time.sleep(5.0)

	# otherwise, reset the total number of consecutive frames and the
	# stop alarm
	else:
		TOTAL_CONSEC = 0
		STOP = False
                GPG.set_led(GPG.LED_BLINKER_LEFT, 0)
                GPG.set_led(GPG.LED_BLINKER_RIGHT, 0)

	# build the label and draw it on the frame
	label = "{}: {:.2f}%".format(label, proba * 100)
	frame = cv2.putText(frame, label, (10, 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
 
	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
 
# do a bit of cleanup
print("[INFO] cleaning up...")
GPG.set_motor_power(GPG.MOTOR_LEFT + GPG.MOTOR_RIGHT, 0)
GPG.set_led(GPG.LED_BLINKER_LEFT, 0)
GPG.set_led(GPG.LED_BLINKER_RIGHT, 0)
cv2.destroyAllWindows()
vs.stop()
