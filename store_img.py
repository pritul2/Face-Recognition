from keras import backend as K
import cv2
import tensorflow as tf
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
import compare_file

def detect(img):
	img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	detector = MTCNN()
	results = detector.detect_faces(img)
	if len(results) == 0 :
		return
	if results[0]['confidence'] >= 0.5:
		x1, y1, width, height = results[0]['box']
		x1, y1 = abs(x1), abs(y1)
		x2, y2 = x1 + width, y1 + height
		if img is None:
			return 
		img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
		img = extract(img,[x1-20,y1-20,x2+20,y2+20])
		return img
	return 

def extract(img,coord):
	img = img[coord[1]:coord[3],coord[0]:coord[2]]
	if img is None:
		print("line 26")
		return
	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
	img = cv2.equalizeHist(img)
	cv2.imwrite("1.png",img)
	return img

def triplet_loss(y_true,y_pred,alpha = 0.3):
	anchor = y_pred[0]
	positive = y_pred[1]
	negative = y_pred[2]
	pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
	neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
	basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
	loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
	return loss

def vid_capture():

	model = load_model('models/model.h5', custom_objects={'triplet_loss': triplet_loss})
	cap = cv2.VideoCapture(0)
	K.set_image_data_format('channels_last')
	
	while cap.isOpened():
		ret,frame = cap.read()
		if ret:
			frame = detect(frame)
			if frame is None:
				continue
			frame = cv2.resize(frame,(100,100))
			if frame is None:
				continue
			cv2.imshow("frame",frame)
			cv2.waitKey(1)
			name = compare_file.compare(model)
			print(name)
			if name is None or name == " " or name == "unknown":
				print("unknown")
		#return frame

vid_capture()