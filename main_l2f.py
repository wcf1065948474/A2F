import argparse
import logging
import sys
import torch
import train
import dataset
import os
import time

parser = argparse.ArgumentParser(description='L2F_Net')
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

if not os.path.exists(opt.checkpoints):
    os.mkdir(opt.checkpoints)
if opt.resume:
    opt.logdir = '{}/{}'.format(opt.checkpoints, opt.resume_name)
else:
    opt.logdir = '{}/{}-{}'.format(opt.checkpoints, opt.idt_name, time.strftime("%Y%m%d-%H%M%S"))
if not os.path.exists(opt.logdir):
    os.mkdir(opt.logdir)

log_format = '%(asctime)s - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(opt.logdir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logger = logging.getLogger()
logger.addHandler(fh)

# for key, val in vars(opt).items():
#     if isinstance(val, list):
#         val = [str(v) for v in val]
#         val = ','.join(val)
#     if val is None:
#         val = 'None'
#     logger.info('{:>20} : {:<50}'.format(key, val))

logger.info('==> Preparing data..')
trainset = dataset.APBDataset(opt.landmark_path, opt.crop_len, opt.idt_name, mode='train', read_img=True)
testset = dataset.APBDataset(opt.landmark_path, opt.crop_len, opt.idt_name, mode='test', read_img=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchsize, shuffle=True, num_workers=8)
testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batchsize, shuffle=True, num_workers=8)

logger.info('==> Building model..')
net = train.TrainFlowNet(opt, logger)

def train(epoch):
    net.run_train(trainloader)


for epoch in range(1, opt.epochs):
    train(epoch)
    # test(epoch)
    logger.info('-' * 50)