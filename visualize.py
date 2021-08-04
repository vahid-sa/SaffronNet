import cv2
from os import path as osp
import numpy as np

from retinanet.utils import ActiveLabelMode
from utils.visutils import draw_line
from retinanet.settings import NAME, X, Y, ALPHA, SCORE, LABEL, TRUTH

active_clor_plate = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 255, 0), 3: (255, 0, 0), 4: (0, 255, 255)}
nms_color_plate = {0: (0, 255, 255), 1: (0, 255, 0)}


def draw(loader, detections, images_dir, output_dir, ext=".jpg"):
	name_col = detections[:, NAME]
	print()
	for i in range(len(loader)):
		img_path = osp.join(images_dir, "{0}{1}".format(loader[i]["name"], ext))
		img = cv2.imread(img_path)
		img_name = float(int(loader[i]["name"]))
		image_detections = detections[name_col == img_name]
		for j in range(len(image_detections)):
			det = image_detections[j]
			im_name, x, y, alpha, score, label, truth = det[[NAME, X, Y, ALPHA, SCORE, LABEL, TRUTH]]
			if truth == 0: # uncertain
				line_color = (0, 0, 255)
			elif truth == 1:
				line_color = (0, 255, 0)
			else:
				line_color = (255, 0, 0)
			center_color = (0, 0, 0)
			img = draw_line(
				image=img,
				p=(x, y),
				alpha=90.0 - alpha,
				line_color=line_color,
				center_color=center_color,
				half_line=True)
		save_path = osp.join(output_dir, loader[i]["name"] + ext)
		cv2.imwrite(save_path, img)
		print("\rsaved {0}/{1}".format(i, len(loader)), end='')
	print()


def draw_noisy_uncertain_gt(loader, detections, images_dir, output_dir, ext=".jpg"):
	name_col = detections[:, NAME]
	print()
	for i in range(len(loader)):
		img_path = osp.join(images_dir, "{0}{1}".format(loader[i]["name"], ext))
		img = cv2.imread(img_path)
		img_name = float(int(loader[i]["name"]))
		image_detections = detections[name_col == img_name]
		hase_uncertain = False
		for j in range(len(image_detections)):
			det = image_detections[j]
			im_name, x, y, alpha = det[[NAME, X, Y, ALPHA]]
			status = det[-1]
			if status == ActiveLabelMode.uncertain.value or status == ActiveLabelMode.corrected.value:
				hase_uncertain = True
			line_color = active_clor_plate[status]
			center_color = (0, 0, 0)
			img = draw_line(
				image=img,
				p=(x, y),
				alpha=90.0 - alpha,
				line_color=line_color,
				center_color=center_color,
				half_line=True,
				line_thickness=3)
		save_path = osp.join(output_dir, loader[i]["name"] + ext)
		if hase_uncertain:
			cv2.imwrite(save_path, img)
		print("\rsaved {0}/{1}".format(i, len(loader)), end='')
	print()


def draw_correct_noisy(loader, detections, images_dir, output_dir, ext=".jpg"):
	name_col = detections[:, NAME]
	print()
	for i in range(len(loader)):
		count_noisy_annots, count_corrected_annots = 0, 0
		img_path = osp.join(images_dir, "{0}{1}".format(loader[i]["name"], ext))
		img = cv2.imread(img_path)
		img_name = float(int(loader[i]["name"]))
		image_detections = detections[name_col == img_name]
		has_uncertain = False
		for j in range(len(image_detections)):
			det = image_detections[j]
			im_name, x, y, alpha = det[[NAME, X, Y, ALPHA]]
			status = det[-1]
			if status == ActiveLabelMode.corrected.value:
				has_uncertain = True
				count_corrected_annots += 1
			elif status ==ActiveLabelMode.noisy.value:
				count_noisy_annots += 1
			else:
				raise ValueError("annotation can be correct or noisy")
			line_color = active_clor_plate[status]
			center_color = (0, 0, 0)
			img = draw_line(
				image=img,
				p=(x, y),
				alpha=90.0 - alpha,
				line_color=line_color,
				center_color=center_color,
				half_line=True,
				line_thickness=3)
		filename = "{0}_{1}_{2}{3}".format(loader[i]["name"], count_corrected_annots, count_noisy_annots, ext)
		save_path = osp.join(output_dir, filename)
		if has_uncertain:
			cv2.imwrite(save_path, img)
		print("\rsaved {0}/{1}".format(i, len(loader)), end='')
	print()


def draw_selected_ignored(loader, detections, images_dir, output_dir, ext=".jpg"):
	main_name_col = detections['main'][:, NAME]
	co_name_col = detections['co'][:, NAME]
	print()
	for i in range(len(loader)):
		img_path = osp.join(images_dir, "{0}{1}".format(loader[i]["name"], ext))
		img = cv2.imread(img_path)
		img_name = float(int(loader[i]["name"]))
		main_detections = detections['main'][main_name_col == img_name]
		main_detections = np.concatenate([main_detections, np.ones(shape=(main_detections.shape[0], 1))], axis=1)
		co_detections = detections['co'][co_name_col == img_name]
		co_detections = np.concatenate([co_detections, np.zeros(shape=(co_detections.shape[0], 1))], axis=1)
		image_detections = np.concatenate([main_detections, co_detections])
		image_detections = image_detections[image_detections[:, -1].argsort()]

		for j in range(len(image_detections)):
			det = image_detections[j]
			im_name, x, y, alpha = det[[NAME, X, Y, ALPHA]]
			status = det[-1]
			line_color = nms_color_plate[status]
			center_color = (0, 0, 0)
			img = draw_line(
				image=img,
				p=(x, y),
				alpha=90.0 - alpha,
				line_color=line_color,
				center_color=center_color,
				half_line=True,
				line_thickness=3)
		filename = str(loader[i]["name"]) + ext
		save_path = osp.join(output_dir, filename)
		cv2.imwrite(save_path, img)
		print("\rsaved {0}/{1}".format(i, len(loader)), end='')
	print()
