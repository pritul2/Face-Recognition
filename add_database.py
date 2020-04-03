from keras import backend as K
import cv2
from mtcnn.mtcnn import MTCNN
import math
from scipy.spatial import distance

def enhance_img(img):
	clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))
	lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	l, a, b = cv2.split(lab)
	l2 = clahe.apply(l)
	lab = cv2.merge((l2,a,b))
	img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
	return img

def calc_angle(l):
	
	x1=l[0][0];y1=l[0][1];x2=l[1][0];y2=l[1][1]
	
	slope = (y2-y1)/(x2-x1)
	angle = math.degrees(math.atan(slope))
	return angle

def align_face(img,results):

	angle = calc_angle([results[0]['keypoints']['left_eye'],results[0]['keypoints']['right_eye']])
	
	eyesCenter = ((results[0]['keypoints']['left_eye'][0] + results[0]['keypoints']['right_eye'][0]) // 2,(results[0]['keypoints']['left_eye'][1] + results[0]['keypoints']['right_eye'][1]) // 2)
	print(eyesCenter)
	
	M = cv2.getRotationMatrix2D(eyesCenter, angle, 1)
	img=cv2.warpAffine(img, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)
	
	return img

def detect(img):
	img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	detector = MTCNN()
	results = detector.detect_faces(img)
	a = results[0]['keypoints']['left_eye']
	b = results[0]['keypoints']['right_eye']
	print(a)
	dst = distance.euclidean(a, b)

	print(dst)
	if len(results) == 0 :
		return
	if results[0]['confidence'] >= 0.5:
		print(results)
		x1, y1, width, height = results[0]['box']
		x1, y1 = abs(x1), abs(y1)
		x2, y2 = x1 + width, y1 + height
		img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		img = cv2.equalizeHist(img)
		img = extract(img,[x1-20,y1-20,x2+20,y2+20])
		#img=align_face(img,results)
		return img
	return 

def extract(img,coord):

	img = img[coord[1]:coord[3],coord[0]:coord[2]]
	cv2.imwrite("1.png",img)
	return img
def equalize(img):
	[r,g,b] = cv2.split(img)
	r = cv2.equalizeHist(r)
	g = cv2.equalizeHist(g)
	b = cv2.equalizeHist(b)
	img = cv2.merge((b,g,r),img)
	cv2.imshow("image",img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return img
def vid_capture():
	#model = loadmodel.load_model()
	cap = cv2.VideoCapture(0)
	#ret,frame = cap.read()
	i=0
	K.set_image_data_format('channels_last')
	while cap.isOpened():
		print("cap is open")
		ret,frame = cap.read()
		fps = cap.get(cv2.CAP_PROP_FPS)
		print(fps)
		#frame = equalize(frame)
		print(ret)
		if ret:
			print("inside ret")
			frame = detect(frame)

			frame = cv2.resize(frame,(100,100),cv2.INTER_LINEAR)
			'''im = Image.open(r"/Users/prituldave/pritul_E_drive/SIH/1.png")
			im3 = ImageEnhance.Color(im)
			im3.enhance(1.5).show()
			#frame = enhance_img(frame)'''
			if frame is None:
				continue
			cv2.imshow("frame",frame)
			cv2.waitKey(1)
			frame = cv2.equalizeHist(frame)
			cv2.imwrite("/Users/prituldave/pritul_E_drive/SIHv4/faces/darshit/"+str(i)+".png",frame)
			i=(i+1)

vid_capture()