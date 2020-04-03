import os
import numpy as np
from fr_utils import img_to_encoding

flag2 = True
ls = []
def compare(model):
	global ls
	print(model)
	embedding1 = img_to_encoding("/Users/prituldave/pritul_E_drive/SIHv5/1.png",model)
	global flag2
	os.chdir("faces")
	x = (os.listdir())
	min_dist = 100.0
	output_name = " "
	for names in x:
		if names == ".DS_Store":
			continue
		os.chdir(names)

		file_images = os.listdir()
		
		sum = 0
		cnt = 0
		for _images in file_images:
			if _images == ".DS_Store":
				continue
			embedding2 = img_to_encoding(_images,model)
			dist = np.linalg.norm(embedding2 - embedding1)
			#sum = sum + dist
			#cnt = cnt + 1

		if dist<min_dist:

			output_name = names
			min_dist = dist

		os.chdir("..")
	os.chdir("..")
	print("recognized person is ",output_name)
	print("min distance is ",min_dist)
	if(min_dist>=0.60):
		output_name = "unknown"
		ls.append(output_name)
		print("unknown")
	else:
		output_name = "known"
		ls.append(output_name)
		print("known")
	#if output_name == "unknown":
	#	print("unknown")
	return output_name