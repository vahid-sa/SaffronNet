import cv2
from os import path as osp
from utils.visutils import draw_line
from retinanet.settings import NAME, X, Y, ALPHA, SCORE, LABEL, TRUTH


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
