import cv2
import os
from scipy.spatial import distance
from mtcnn.mtcnn import MTCNN
ls = os.listdir("/Users/prituldave/pritul_E_drive/SIHv4/faces/pritul")
for i in ls:
	print(i)
	if i ==".DS_Store":
		continue
	img = cv2.imread("/Users/prituldave/pritul_E_drive/SIHv4/faces/pritul/"+i)
	detector = MTCNN()
	results = detector.detect_faces(img)
	a = results[0]['keypoints']['left_eye']
	b = results[0]['keypoints']['right_eye']
	print(a)
	dst = distance.euclidean(a, b)
	print(dst)