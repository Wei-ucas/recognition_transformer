import os
import numpy as np
from data_tools.data_utils import get_vocabulary
import scipy.io as sio
import re
from itertools import chain
import math
# import cv2
from PIL import Image

class SynthTextLoader(object):
	def __init__(self, data_dir, gt_mat_path, max_len=25):
		super(SynthTextLoader, self).__init__()
		self.data_dir = data_dir
		self.mat_contents = sio.loadmat(gt_mat_path)
		self.images_name = self.mat_contents['imnames'][0]
		self.num_samples = len(self.images_name)
		self.max_len = max_len
		self.voc, self.char2id, _ = get_vocabulary("ALLCASES_SYMBOLS")

	def get_sample(self, index):

		text_polys = []
		text_tags = []
		labels = []
		words = []
		label_masks = []

		image_name = self.images_name[index][0]
		# img = cv2.imread(os.path.join(self.data_dir, image_name), cv2.IMREAD_COLOR)
		img = Image.open(os.path.join(self.data_dir, image_name)).convert('RGB')
		img_width, img_height = img.size

		txt = self.mat_contents['txt'][0][index]
		txt = [re.split(' \n|\n |\n| ', t.strip()) for t in txt]
		txt = list(chain(*txt))
		txt = [t for t in txt if len(t) > 0]

		# Validation
		if len(np.shape(self.mat_contents['wordBB'][0][index])) == 2:
			wordBBlen = 1
		else:
			wordBBlen = self.mat_contents['wordBB'][0][index].shape[-1]

		if wordBBlen == len(txt):
			# Crop image and save
			for word_indx in range(len(txt)):
				if len(np.shape(self.mat_contents['wordBB'][0][index])) == 2:  # only one word (2,4)
					wordBB = self.mat_contents['wordBB'][0][index]
				else:  # many words (2,4,num_words)
					wordBB = self.mat_contents['wordBB'][0][index][:, :, word_indx]

				if np.shape(wordBB) != (2, 4):
					continue

				pts1 = np.float32([[wordBB[0][0], wordBB[1][0]],
								   [wordBB[0][1], wordBB[1][1]],
								   [wordBB[0][2], wordBB[1][2]],
								   [wordBB[0][3], wordBB[1][3]]])

				height = math.sqrt((wordBB[0][0] - wordBB[0][3]) ** 2 + (wordBB[1][0] - wordBB[1][3]) ** 2)
				width = math.sqrt((wordBB[0][0] - wordBB[0][1]) ** 2 + (wordBB[1][0] - wordBB[1][1]) ** 2)

				# Coord validation check
				if (height * width) <= 0:
					continue
				elif (height * width) > (img_height * img_width):
					continue
				else:
					valid = True
					for i in range(2):
						for j in range(4):
							if wordBB[i][j] < 0 or wordBB[i][j] > img.size[i]:
								valid = False
								break
						if not valid:
							break
					if not valid:
						continue

				# Add the polygons, tags and transcription to list
				text_polys.append(pts1)
				text_tags.append(False)

				label = np.full((self.max_len), self.char2id['PAD'], dtype=np.int)
				label_mask = np.full((self.max_len), 0, dtype=np.int)
				label_list = []
				for char in txt[word_indx]:
					if char in self.char2id:
						label_list.append(self.char2id[char])
					else:
						label_list.append(self.char2id['UNK'])

				if len(label_list) > (self.max_len - 1):
					label_list = label_list[:(self.max_len - 1)]
				label_list = label_list + [self.char2id['EOS']]
				label[:len(label_list)] = np.array(label_list)
				label_len = len(label_list)
				label_mask[:label_len] = 1
				labels.append(label)
				words.append(txt[word_indx])
				label_masks.append(label_mask)
			assert len(text_polys) == len(text_tags) == len(labels) == len(label_masks) == len(words)
			if len(labels) == 0:
				pass
			return img, image_name, np.array(text_polys), np.array(text_tags), np.array(labels), np.array(label_masks), words

		else:
			return None, None, None, None, None

if __name__ == "__main__":
	stl = SynthTextLoader(data_dir="/data2/data/SynthText/", gt_mat_path="/data2/data/SynthText/gt.mat")
	print("Load dataset done!")
	for i in range(100):
		img, img_name, polygons, _, _ = stl.get_sample(i)
		polygons = polygons.astype(np.int32)
		img_vis = img.copy()
		for p in polygons:
			img_vis = cv2.polylines(img_vis, [p.reshape((-1, 1, 2))], True, (0, 0, 255), 1)

		cv2.imwrite(os.path.join("syntext_poly_vis", "{}_{}.jpg".format("img", os.path.basename(img_name))), img_vis)
