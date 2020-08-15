import random
from torch.utils.data import dataset
import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np


class APBDataset(dataset.Dataset):
	def __init__(self, root, crop_len, idt_name='man1', mode='train', img_size=256, read_img=False):
		self.root = root
		self.read_img = read_img
		self.idt_name = idt_name
		self.crop_len = crop_len
		if not isinstance(mode, list):
			mode = [mode]

		self.transforms_image = transforms.Compose([transforms.ToTensor(),
				transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])

		self.data_all = list()
		for m in mode:
			training_data_path = os.path.join(self.root, self.idt_name, '{}_{}.t7'.format(img_size, m))
			training_data = torch.load(training_data_path)
			img_paths = training_data['img_paths']
			audio_features = training_data['audio_features']
			lands = training_data['lands']
			poses = training_data['poses']
			eyes = training_data['eyes']
			for i in range(len(img_paths)):
				img_path = [os.path.join(self.root, self.idt_name, p) for p in img_paths[i]]  # [image, landmark]
				audio_feature = audio_features[i]
				land = lands[i]
				pose = poses[i]
				eye = eyes[i]
				self.data_all.append([img_path, audio_feature, land, pose, eye])
		self.data_all.sort(key=lambda x: int(x[0][0].split('/')[-1].split('.')[0]))

	def __len__(self):
		return len(self.data_all)-self.crop_len

	def __getitem__(self, index):
		audio_list = []
		img1_list = []
		img2_list = []
		lab1_list = []
		lab2_list = []
		poseeye1_list = []
		poseeye2_list = []
		landA1_list = []
		landA2_list = []
		for i in range(self.crop_len):
			img_path_A1, audio_feature_A1, land_A1, pose_A1, eye_A1 = self.data_all[index+i]
			img_path_A2, audio_feature_A2, land_A2, pose_A2, eye_A2 = random.sample(self.data_all, 1)[0]
			# audio
			audio_feature_A1 = torch.tensor(audio_feature_A1).unsqueeze(dim=0)
			audio_list.append(audio_feature_A1)
			# if self.read_img:
			img1 = Image.open(img_path_A1[0]).convert('RGB')
			img1 = self.transforms_image(img1)
			img2 = Image.open(img_path_A2[0]).convert('RGB')
			img2 = self.transforms_image(img2)
			lab1 = Image.open(img_path_A1[1]).convert('RGB')
			lab1 = self.transforms_image(lab1)
			lab2 = Image.open(img_path_A2[1]).convert('RGB')
			lab2 = self.transforms_image(lab2)

			img1_list.append(img1)
			img2_list.append(img2)
			lab1_list.append(lab1)
			lab2_list.append(lab2)
			# else:
				# pose
			pose_A1 = torch.tensor(pose_A1)
			pose_A2 = torch.tensor(pose_A2)
			# eye
			eye_A1 = torch.tensor(eye_A1)
			eye_A2 = torch.tensor(eye_A2)
			# landmark
			land_A1 = torch.tensor(land_A1)
			land_A2 = torch.tensor(land_A2)
			poseeye1_list.append(torch.cat([pose_A1,eye_A1],0))
			poseeye2_list.append(torch.cat([pose_A2,eye_A2],0))
			landA1_list.append(land_A1)
			landA2_list.append(land_A2)
		
		if self.read_img:
			img1 = torch.stack(img1_list,0)
			img2 = torch.stack(img2_list,0)
			lab1 = torch.stack(lab1_list,0)
			lab2 = torch.stack(lab2_list,0)
			audio_feature_A1 = torch.cat(audio_list,0)
			poseeye_A1 = torch.stack(poseeye1_list,0)
			poseeye_A2 = torch.stack(poseeye2_list,0)
			landA1 = torch.stack(landA1_list,0)
			landA2 = torch.stack(landA2_list,0)
			return {'img1':img1,'img2':img2,'lab1':lab1,'lab2':lab2,'audio':audio_feature_A1,
			'poseeye_A1':poseeye_A1,'poseeye_A2':poseeye_A2,'landA1':landA1,'landA2':landA2}
		else:
			audio_feature_A1 = audio_list[0]
			poseeye_A1 = torch.cat([pose_A1,eye_A1],0)
			poseeye_A2 = torch.cat([pose_A2,eye_A2],0)
			land_A1 = landA1_list[0]
			land_A2 = landA2_list[0]
			return [audio_feature_A1, poseeye_A1, poseeye_A2], [land_A1, land_A2]




class L2FaceDataset(dataset.Dataset):
    def __init__(self, opt):
        img_size = opt.img_size
        root = '../AnnVI/feature/{}'.format(opt.name.split('_')[0])
        image_dir = '{}/{}_image_crop'.format(root, img_size)
        label_dir = '{}/{}_landmark_crop_thin'.format(root, img_size)
        # label_dir = '{}/512_landmark_crop'.format(root)
        self.labels = []

        imgs = os.listdir(image_dir)
        # if 'man' in opt.name:
        #     imgs.sort(key=lambda x:int(x.split('.')[0]))
        # else:
        #     imgs.sort(key=lambda x: (int(x.split('.')[0].split('-')[0]), int(x.split('.')[0].split('-')[1])))
        for img in imgs:
            img_path = os.path.join(image_dir, img)
            lab_path = os.path.join(label_dir, img)
            if os.path.exists(lab_path):
                self.labels.append([img_path, lab_path])
        # transforms.Resize([img_size, img_size], Image.BICUBIC),
        self.transforms_image = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # transforms.Resize([img_size, img_size], Image.BICUBIC),
        self.transforms_label = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.labels)


    def __getitem__(self, index):
        img_path, lab_path = self.labels[index]
        img = Image.open(img_path).convert('RGB')
        lab = Image.open(lab_path).convert('RGB')
        img = self.transforms_image(img)
        lab = self.transforms_label(lab)

        imgA_path, labA_path = random.sample(self.labels, 1)[0]
        imgA = Image.open(imgA_path).convert('RGB')
        imgA = self.transforms_image(imgA)


        return {'A': imgA, 'A_label': lab, 'B': img, 'B_label': lab}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.labels)