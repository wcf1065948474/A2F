import os
import cv2
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import sys
import torch
import models
import dataset


parser = argparse.ArgumentParser(description='A2F_Net')
parser.add_argument('--isTrain', default=True, type=bool, help='running mode')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--lrd', default=0.00001, type=float, help='learning rate')
parser.add_argument('--every', default=300, type=float, help='learning rate decay')
parser.add_argument('--gpus', default='0', type=str, help='gpus')
parser.add_argument('--landmark_path', default='../AnnVI/feature', type=str, help='landmark path that contains several persons')
parser.add_argument('--checkpoints', default='checkpoints', type=str, help='checkpoint path')
parser.add_argument('--epochs', default=500, type=int, help='epochs')
parser.add_argument('--resume', '-r', default=False, type=bool, help='resume')
parser.add_argument('--resume_epoch', default=None, type=int, help='resume epoch')
parser.add_argument('--resume_name', default='L2F', type=str, help='resume epoch')
parser.add_argument('--idt_name', default='man1', type=str, help='identity name')
parser.add_argument('--results_dir',default='result')
parser.add_argument('--image_nc',default=3)
parser.add_argument('--structure_nc',default=3)
parser.add_argument('--frames_D_V',default=3)
parser.add_argument('--gan_mode',default='lsgan')
parser.add_argument('--kernel_size',default={'2':5,'3':3})
parser.add_argument('--ratio_g2d',default=0.1)
parser.add_argument('--batchsize',default=2)
parser.add_argument('--crop_len',default=4)
parser.add_argument('--lambda_style',default=500.0)
parser.add_argument('--lambda_content',default=0.5)
parser.add_argument('--lambda_correct',default=5.0)
parser.add_argument('--lambda_g',default=2.0)
parser.add_argument('--lambda_rec',default=5.0)
parser.add_argument('--lambda_regularization',default=0.0025)
parser.add_argument('--attn_layer',default=[2, 3])

opt = parser.parse_args()


def drawCircle(img, shape, radius=1, color=(255, 255, 255), thickness=1):
    for p in shape:
        img = cv2.circle(img, (int(p[0]), int(p[1])), radius, color, thickness)
    return img


def vector2points(landmark):
    shape = []
    for i in range(len(landmark) // 2):
        shape.append([landmark[2 * i], landmark[2 * i + 1]])
    return shape

def load_networks(model,netnames,path='weight/'):
    if netnames is str:
        if os.path.exists(path+netnames+'.pth'):
            model.load_state_dict(torch.load(path+netnames+'.pth'))
            model.evel()
            model.cuda()
            print('load {}'.format(netnames))
        else:
            print('{} does not find'.format(netnames))
            exit()
    for netname in netnames:
        net = getattr(model,netname)
        if os.path.exists(path+netname+'.pth'):
            net.load_state_dict(torch.load(path+netname+'.pth'))
            net.evel()
            net.cuda()
            print('load {}'.format(netname))
        else:
            print('{} does not find'.format(netname))
            exit()

opt.isTrain = False
landmark2face = models.FaceGenerator()
audio2landmark = models.APBNet()
load_networks(landmark2face,['source_previous','source_reference','target','flow_net'])
load_networks(audio2landmark,'APBNet')
transforms_label = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# dataset
feature_path = '../AnnVI/feature'
idt_name = 'man1'
testset = dataset.APBDataset(opt.landmark_path, opt.crop_len, opt.idt_name, mode='test', read_img=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batchsize, shuffle=True, num_workers=8)

out_path = 'result'
if not os.path.exists(out_path):
    os.mkdir(out_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(os.path.join(out_path, '{}.avi'.format(idt_name)), fourcc, 25.0, (256 * 2, 256))

for idx, data in enumerate(dataloader):
    audio_feature_A1, pose_A1, eye_A1 = data[0][0].cuda(), \
                                        data[0][1].cuda(), \
                                        data[0][2].cuda()
    landmark_A1, landmark_A2 = data[1][0].cuda(),\
                                data[1][1].cuda()

    image_path_A1 = data[2][0][0][0]
    print('\r{}/{}'.format(idx+1, len(dataloader)), end='')

    landmark = audio_net(audio_feature_A1, pose_A1, eye_A1)
    landmark = landmark.cpu().data.numpy().tolist()[0]
    lab_template = np.zeros((256, 256, 3)).astype(np.uint8)
    lab = drawCircle(lab_template.copy(), vector2points(landmark), radius=1, color=(255, 255, 255), thickness=4)
    lab = Image.fromarray(lab).convert('RGB')
    lab = transforms_label(lab).unsqueeze(0)

    input_data = {'A': lab, 'A_label': lab, 'B': lab, 'B_label': lab}
    model.set_input(input_data)
    model.test()
    visuals = model.get_current_visuals()
    B_img_f = tensor2im(visuals['fake_B'])
    B_img = cv2.imread(image_path_A1)
    B_img = cv2.cvtColor(B_img, cv2.COLOR_BGR2RGB)
    B_img = cv2.resize(B_img, (256, 256))

    img_out = np.concatenate([B_img_f, B_img], axis=1)
    for _ in range(5):  # five times slower
        out.write(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
    if idx == 100:
        break
out.release()
