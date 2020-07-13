import os
import glob
import numpy as np
from PIL import Image

from data_tools.data_utils import get_vocabulary

class ICDAR15Loader(object):
	def __init__(self, data_dir, gt_dir, with_script=False, shuffle=False, max_len=25):
		super(ICDAR15Loader, self).__init__()
		self.data_dir = data_dir
		self.images_path = self.get_images()
		self.num_samples = len(self.images_path)
		self.gt_dir = gt_dir
		self.max_len = max_len
		self.with_script = with_script
		self.shuffle = shuffle # shuffle the polygons
		self.voc, self.char2id, _ = get_vocabulary("ALLCASES_SYMBOLS")

	def get_images(self):
		files = []
		for ext in ['jpg', 'png', 'jpeg', 'JPG']:
			files.extend(glob.glob(os.path.join(self.data_dir, '*.{}'.format(ext))))
		return files

	def map2labelfile(self, img_path):
		img_name = os.path.basename(img_path)
		gt_name = "gt_{}.txt".format(img_name.split(".")[0])
		if self.gt_dir is not None:
			gt_path = os.path.join(self.gt_dir, gt_name)
		else:
			gt_path = gt_name
		return gt_path

	def get_sample(self, index):
		img_path = self.images_path[index]
		gt_path = self.map2labelfile(img_path)
		img = Image.open(os.path.join(self.data_dir, img_path)).convert('RGB')

		text_polys = []
		text_tags = []
		labels = []
		words = []
		label_masks = []
		if gt_path is None:
			return img, img_path, None, None, None
		else:
			if not os.path.exists(gt_path):
				print("{} not exists!".format(gt_path))
				return img, img_path, None, None, None
		with open(gt_path, 'r', encoding="utf-8-sig") as f:
			for line in f.readlines():
				line = line.replace('\xef\xbb\bf', '')
				line = line.replace('\xe2\x80\x8d', '')
				line = line.strip()
				line = line.split(',')
				if self.with_script:
					line.pop(8)  # since icdar17 has script
				# Deal with transcription containing ,
				if len(line) > 9:
					word = ",".join([p for p in line[8:]])
				else:
					word = line[-1]

				temp_line = list(map(eval, line[:8]))
				x1, y1, x2, y2, x3, y3, x4, y4 = map(float, temp_line)

				if word == '*' or word == '###' or word == '':
					continue

				text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

				label = np.full((self.max_len), self.char2id['PAD'], dtype=np.int)
				label_mask = np.full((self.max_len), 0, dtype=np.int)
				label_list = []
				for char in word:
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
				words.append(word)
				label_masks.append(label_mask)
				text_tags.append(False)
		assert len(text_polys) == len(text_tags) == len(labels) == len(label_masks) == len(words)

		return img, img_path, np.array(text_polys), np.array(text_tags), np.array(labels), np.array(label_masks), words