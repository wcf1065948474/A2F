import cv2
import glob
import numpy as np
import os
import time
import logging
import itertools
import torch
import torch.nn as nn
from PIL import Image
import models
from utils import *
from loss import *
import external_function
from torch.optim import lr_scheduler
from collections import OrderedDict

def _unfreeze(*args):
    """ unfreeze the network for parameter update"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = True

def _freeze(*args):
    """freeze the network for forward process"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = False


def get_scheduler(optimizer, opt):
    """Get the training learning rate for different epoch"""
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch+1+1+opt.iter_count-opt.niter) / float(opt.niter_decay+1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'exponent':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        raise NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler



class TrainAudioToLandmark(object):

    def __init__(self, opt, logger):
        self.opt = opt
        self.isTrain = opt.isTrain
        self.lr = opt.lr
        self.every = opt.every
        self.epoch = 0
        self.best_loss = float("inf")
        self.idt_name = opt.idt_name
        self.logdir = opt.logdir
        self.logger = logger
        # loss
        self.criterionGAN = GANLoss(gan_mode='mse').cuda()
        self.criterionL1 = nn.L1Loss()
        # G
        self.netG = models.APBNet()
        self.netG.apply(weight_init)
        if opt.resume:
            checkpoint = torch.load('{}/{}.pth'.format(self.logdir, opt.resume_epoch if opt.resume_epoch else '{}_best'.format(self.idt_name)))
            self.netG.load_state_dict(checkpoint['net_G'])
            self.epoch = checkpoint['epoch']
        self.netG.cuda()
        # D
        if self.isTrain:  # define discriminators
            self.netD = models.Discriminator()
            self.netD.apply(weight_init)
            if opt.resume:
                self.netD.load_state_dict(checkpoint['net_D'])
            self.netD.cuda()

            self.netD_poseye = models.Discriminator_poseeye()
            self.netD_poseye.apply(weight_init)
            if opt.resume:
                self.netD_poseye.load_state_dict(checkpoint['net_D_poseeye'])
            self.netD_poseye.cuda()

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.lr, betas=(0.99, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.lr, betas=(0.99, 0.999))
            self.optimizer_D_poseeye = torch.optim.Adam(self.netD_poseye.parameters(), lr=self.lr, betas=(0.99, 0.999))

    def train(self):
        self.isTrain = True

    def eval(self):
        self.isTrain = False

    def reset(self):
        self.loss_log_L1 = 0
        self.loss_log_G_A = 0

        self.loss_log_D_A_F = 0
        self.loss_log_D_A_T = 0

    def test_draw(self, dataloader):
        def drawCircle(img, shape, radius=1, color=(255, 255, 255), thickness=1):
            for i in range(len(shape) // 2):
                img = cv2.circle(img, (int(shape[2 * i]), int(shape[2 * i + 1])), radius, color, thickness)
            return img

        def drawArrow(img, shape1, shape2, ):
            for i in range(len(shape1) // 2):
                point1 = (int(shape1[2 * i]), int(shape1[2 * i + 1]))
                point2 = (int(shape1[2 * i] + shape2[2 * i]), int(shape1[2 * i + 1] + shape2[2 * i + 1]))
                img = cv2.circle(img, point2, radius=6, color=(0, 0, 255), thickness=2)
                img = cv2.line(img, point1, point2, (255, 255, 255), thickness=2)
            return img

        root = self.logdir
        s_pathA = '{}/resultA'.format(root)
        s_pathB = '{}/resultB'.format(root)
        if not os.path.exists(s_pathA):
            os.mkdir(s_pathA)
        if not os.path.exists(s_pathB):
            os.mkdir(s_pathB)
        with torch.no_grad():
            for batch_idx, data in enumerate(dataloader):
                self.set_input(data)
                self.forward()
                img_size = 256
                img_template = np.zeros((img_size, img_size, 3))
                img_fake_A1 = drawCircle(img_template.copy(), self.fake_A.squeeze(0).data, radius=1,
                                         color=(255, 255, 255), thickness=2)
                img_A1 = drawCircle(img_template.copy(), self.land_A1.squeeze(0).data, radius=1,
                                    color=(255, 255, 255), thickness=2)
                img_fake_B1 = drawCircle(img_template.copy(), self.fake_B.squeeze(0).data, radius=1,
                                         color=(255, 255, 255), thickness=2)
                img_B1 = drawCircle(img_template.copy(), self.land_B1.squeeze(0).data, radius=1,
                                    color=(255, 255, 255), thickness=2)

                img_compareA = np.concatenate([img_template[:, :, 0][:, :, np.newaxis], img_fake_A1[:, :, 0][:, :, np.newaxis],
                                               img_A1[:, :, 0][:, :, np.newaxis]], axis=2)
                img_compareB = np.concatenate([img_template[:, :, 0][:, :, np.newaxis], img_fake_B1[:, :, 0][:, :, np.newaxis],
                                               img_A1[:, :, 2][:, :, np.newaxis]], axis=2)
                cv2.imwrite('{}/{}.jpg'.format(s_pathA, batch_idx), img_compareA)
                cv2.imwrite('{}/{}.jpg'.format(s_pathB, batch_idx), img_compareB)
                print('\r{}'.format(batch_idx + 1), end='')

    def run_train(self, dataloader, epoch=None):
        self.epoch += 1
        if epoch:
            self.epoch = epoch
        self.reset()
        adjust_learning_rate(self.optimizer_G, self.lr, self.epoch, every=self.every)
        adjust_learning_rate(self.optimizer_D, self.lr, self.epoch, every=self.every)
        for batch_idx, train_data in enumerate(dataloader):
            self.batch_idx = batch_idx + 1
            self.set_input(train_data)
            self.optimize_parameters()
            log_string = 'train\t -> '
            log_string += 'epoch {:>3} '.format(self.epoch)
            log_string += 'batch {:>4} '.format(batch_idx + 1)
            log_string += '|loss_L1 {:.5f}'.format(self.loss_log_L1 / (batch_idx + 1))
            log_string += '|loss_G_A {:.5f}'.format(self.loss_log_G_A / (batch_idx + 1))
            log_string += '|loss_D_A_F {:.5f}'.format(self.loss_log_D_A_F / (batch_idx + 1))
            log_string += '|loss_D_A_T {:.5f}'.format(self.loss_log_D_A_T / (batch_idx + 1))
            print('\r' + log_string, end='')
        print('\r', end='')
        self.logger.info(log_string)

    def run_test(self, dataloader, epoch=None):
        if epoch:
            self.epoch = epoch
        self.reset()
        for batch_idx, test_data in enumerate(dataloader):
            self.batch_idx = batch_idx + 1
            self.set_input(test_data)
            self.evaluate_loss()
            log_string = 'test\t -> '
            log_string += 'epoch {:>3} '.format(self.epoch)
            log_string += 'batch {:>4} '.format(batch_idx + 1)
            log_string += '|loss_L1 {:.5f}'.format(self.loss_log_L1 / (batch_idx + 1))
            log_string += '|loss_G_A {:.5f}'.format(self.loss_log_G_A / (batch_idx + 1))
            log_string += '|loss_D_A_F {:.5f}'.format(self.loss_log_D_A_F / (batch_idx + 1))
            log_string += '|loss_D_A_T {:.5f}'.format(self.loss_log_D_A_T / (batch_idx + 1))
            print('\r'+log_string, end='')
        print('\r', end='')
        self.logger.info(log_string)
        if self.loss_log_L1 / self.batch_idx < self.best_loss and not self.isTrain:
            self.best_loss = self.loss_log_L1 / self.batch_idx
            self.logger.info('save_best {:.5f}'.format(self.best_loss))
            self.save(mode='best')
        if self.epoch % 50 == 0:
            self.logger.info('save_epoch {:d}'.format(self.epoch))
            self.save(mode=self.epoch)

    def set_input(self, training_data):
        self.z = np.random.uniform(-1.,1.,(self.opt.batchsize*self.opt.crop_len,64)).astype(np.float32)
        self.z = torch.from_numpy(self.z)
        self.z = self.z.cuda()
        self.audio_feature_A1 = training_data['audio'].cuda()
        self.land_A1, self.land_A2 = training_data['landA1'].reshape(-1,212).cuda(),training_data['landA2'].reshape(-1,212).cuda()
        self.pose_eye_A1 = training_data['poseeye_A1'].reshape(-1,5).cuda()
        self.pose_eye_A2 = training_data['poseeye_A2'].reshape(-1,5).cuda()

    def optimize_parameters(self):
        self.forward()
        # G
        self.set_requires_grad([self.netD,self.netD_poseye], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D
        if self.batch_idx % 1 == 0:
            self.set_requires_grad([self.netD,self.netD_poseye], True)
            self.optimizer_D.zero_grad()
            self.optimizer_D_poseeye.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
            self.optimizer_D_poseeye.step()

    def evaluate_loss(self):
        self.forward()
        # G
        self.loss_L1 = self.criterionL1(self.fake_A, self.land_A1)
        self.loss_G_A = self.criterionGAN(self.netD(self.fake_A, self.land_A2), True)
        self.loss_log_L1 += self.loss_L1.item()
        self.loss_log_G_A += self.loss_G_A.item()
        # D
        loss_D_A_F = self.criterionGAN(self.netD(self.fake_A.detach(), self.land_A2.detach()), False)
        loss_D_A_T = self.criterionGAN(self.netD(self.land_A1.detach(), self.land_A2.detach()), True)
        self.loss_log_D_A_F += loss_D_A_F.item()
        self.loss_log_D_A_T += loss_D_A_T.item()

    def forward(self):
        self.fake_A,_,self.fake_pose_eye = self.netG(self.audio_feature_A1, self.z)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def backward_G(self):
        lambda_L1 = 100
        lambda_gan = 0.1

        self.loss_L1 = self.criterionL1(self.fake_A, self.land_A1)+self.criterionL1(self.fake_pose_eye, self.pose_eye_A1)
        self.loss_G_A = self.criterionGAN(self.netD(self.fake_A, self.land_A2), True)+self.criterionGAN(self.netD_poseye(self.fake_pose_eye,self.pose_eye_A2),True)

        self.loss_G = self.loss_L1 * lambda_L1 + self.loss_G_A * lambda_gan
        self.loss_G.backward()
        # log
        self.loss_log_L1 += self.loss_L1.item()
        self.loss_log_G_A += self.loss_G_A.item()

    def backward_D(self):
        lambda_D = 0.1
        loss_D_A_F = self.criterionGAN(self.netD(self.fake_A.detach(), self.land_A2.detach()), False)+\
            self.criterionGAN(self.netD_poseye(self.fake_pose_eye.detach(),self.pose_eye_A2.detach()),False)
        loss_D_A_T = self.criterionGAN(self.netD(self.land_A1.detach(), self.land_A2.detach()), True)+\
            self.criterionGAN(self.netD_poseye(self.pose_eye_A1.detach(), self.pose_eye_A2.detach()), True)
        # Combined loss and calculate gradients
        loss_D = (loss_D_A_F + loss_D_A_T) * 0.5 * lambda_D
        loss_D.backward()
        # log
        self.loss_log_D_A_F += loss_D_A_F.item()
        self.loss_log_D_A_T += loss_D_A_T.item()

    def save(self, mode=None):
        state = {
            'net_G': self.netG.state_dict(),
            'net_D': self.netD.state_dict(),
            'net_D_poseeye': self.netD_poseye.state_dict(),
            'epoch': self.epoch,
        }
        torch.save(state, '{}/{}.pth'.format(self.logdir, '{}_{}'.format(self.idt_name, mode if mode else self.epoch)))


class TrainFlowNet(object):
    def __init__(self, opt, logger):
        self.opt = opt
        self.epoch = 0
        self.opt.which_iter = 0
        self.opt.attn_layer = [2,3]
        self.logger = logger
        self.opt.lr_policy = 'lambda'
        self.opt.iter_count = 1
        self.opt.niter = 50000000
        self.opt.niter_decay = 0
        self.opt.lambda_correct = 20.0
        self.opt.lambda_regularization = 0.01
        self.loss_names = ['correctness', 'regularization']
        self.visual_names = ['img1','img2', 'warp', 'flow_fields','masks']
        self.model_names = ['G']
        self.optimizers = []

        self.net_G = models.FlowNet()
        self.audio2landmark = models.APBNet()
        # self.flow2color = util.flow2color()

        if self.opt.isTrain:
            # define the loss functions
            self.Correctness = external_function.PerceptualCorrectness().cuda()
            self.Regularization = external_function.MultiAffineRegularizationLoss(kz_dic={2:5,3:3}).cuda()
            # define the optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, self.net_G.parameters())),
                                                lr=opt.lr, betas=(0.0, 0.999))
            self.optimizers.append(self.optimizer_G)
        # load the pretrained model and schedulers

        """Load networks, create schedulers"""
        
        # if self.opt.isTrain:
        #     self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]

        self.load_networks()

    def load_networks(self):
        weight = torch.load('weight/audio2landmark.pth',map_location='cuda:0')
        weight = weight['net_G']
        self.audio2landmark.load_state_dict(weight)
        self.audio2landmark.eval()
        self.audio2landmark.cuda()
        print('load audio2landmark')


        if os.path.exists('weight/flow_net.pth'):
            self.net_G.load_state_dict(torch.load('weight/flow_net.pth'))
            self.net_G.cuda()
            print('load flow_net')
        else:
            print('flow_net does not find')
            self.net_G.cuda()

    def set_input(self, input):
        # move to GPU and change data types
        self.input = input
        img1,img2,landmark1,landmark2,audio = input['img1'], input['img2'], input['lab1'], input['lab2'], input['audio']

        self.img1 = img1.cuda()
        self.img2 = img2.cuda()
        self.landmark1 = landmark1.cuda()
        self.landmark2 = landmark2.cuda()
        self.audio = audio.cuda()

    def forward(self):
        """Run forward processing to get the inputs"""
        with torch.no_grad():
            audiofeature = self.audio2landmark(self.audio)
        self.flow_fields, self.masks = self.net_G(self.img2,self.landmark2,self.landmark1,audiofeature)
        # self.warp  = self.visi(self.flow_fields[-1])

    def visi(self):
        flow_field = self.flow_fields[-1]
        [b,_,h,w] = flow_field.size()

        source_copy = torch.nn.functional.interpolate(self.img2, (h,w))

        x = torch.arange(w).view(1, -1).expand(h, -1).float()
        y = torch.arange(h).view(-1, 1).expand(-1, w).float()
        x = 2*x/(w-1)-1
        y = 2*y/(h-1)-1
        grid = torch.stack([x,y], dim=0).float().cuda()
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
        flow_x = (2*flow_field[:,0,:,:]/(w-1)).view(b,1,h,w)
        flow_y = (2*flow_field[:,1,:,:]/(h-1)).view(b,1,h,w)
        flow = torch.cat((flow_x,flow_y), 1)

        grid = (grid+flow).permute(0, 2, 3, 1)
        warp = torch.nn.functional.grid_sample(source_copy, grid)
        return  warp


    def backward_G(self):
        """Calculate training loss for the generator"""
        self.correctness = self.Correctness(self.img1, self.img2, self.flow_fields, self.opt.attn_layer)
        self.loss_correctness = self.correctness * self.opt.lambda_correct

        self.regularization = self.Regularization(self.flow_fields)
        self.loss_regularization = self.regularization * self.opt.lambda_regularization

        total_loss = 0
        for name in self.loss_names:
            total_loss += getattr(self, "loss_" + name)
        total_loss.backward()


    def optimize_parameters(self):
        """update netowrk weights"""
        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()


    def run_train(self, dataloader, epoch=None):
        self.epoch += 1
        if epoch:
            self.epoch = epoch
        # self.reset()
        adjust_learning_rate(self.optimizer_G, self.opt.lr, self.epoch, every=self.opt.every)
        # adjust_learning_rate(self.optimizer_D, self.lr, self.epoch, every=self.every)
        for batch_idx, train_data in enumerate(dataloader):
            self.set_input(train_data)
            self.optimize_parameters()
            log_string = 'train\t -> '
            log_string += 'epoch {:>3} '.format(self.epoch)
            log_string += 'batch {:>4} '.format(batch_idx + 1)
            log_string += '|loss_correctness {:.5f}'.format(self.correctness.item())
            log_string += '|loss_regularization {:.5f}'.format(self.regularization.item())
            print('\r' + log_string, end='')
        print('\r', end='')
        self.logger.info(log_string)

    def run_test(self, dataloader, epoch=None):
        if epoch:
            self.epoch = epoch
        self.reset()
        for batch_idx, test_data in enumerate(dataloader):
            self.batch_idx = batch_idx + 1
            self.set_input(test_data)
            self.evaluate_loss()
            log_string = 'test\t -> '
            log_string += 'epoch {:>3} '.format(self.epoch)
            log_string += 'batch {:>4} '.format(batch_idx + 1)
            log_string += '|loss_L1 {:.5f}'.format(self.loss_log_L1 / (batch_idx + 1))
            log_string += '|loss_G_A {:.5f}'.format(self.loss_log_G_A / (batch_idx + 1))
            log_string += '|loss_D_A_F {:.5f}'.format(self.loss_log_D_A_F / (batch_idx + 1))
            log_string += '|loss_D_A_T {:.5f}'.format(self.loss_log_D_A_T / (batch_idx + 1))
            print('\r'+log_string, end='')
        print('\r', end='')
        self.logger.info(log_string)
        if self.loss_log_L1 / self.batch_idx < self.best_loss and not self.isTrain:
            self.best_loss = self.loss_log_L1 / self.batch_idx
            self.logger.info('save_best {:.5f}'.format(self.best_loss))
            self.save(mode='best')
        if self.epoch % 50 == 0:
            self.logger.info('save_epoch {:d}'.format(self.epoch))
            self.save(mode=self.epoch)

    def show(self):
      h=64
      w=64
      img1 = torch.nn.functional.interpolate(self.img1, (h,w))
      img2 = torch.nn.functional.interpolate(self.img2, (h,w))
      landmark1 = torch.nn.functional.interpolate(self.landmark1, (h,w))
      landmark2 = torch.nn.functional.interpolate(self.landmark2, (h,w))
      tmp = torch.cat((img1,img2,landmark1,landmark2,self.warp),3)
      tmp = tmp[0].permute(1,2,0).cpu().detach().numpy()
      tmp = (tmp+1)/2*255
      tmp = Image.fromarray(tmp.astype('uint8'))
      tmp.save('tmp.jpg')



class TrainFace(object):
    def name(self):
        return "Face Image Animation"

    def __init__(self, opt, logger):
        self.opt = opt
        self.logger = logger
        self.epoch = 0
        self.optimizers = []
        self.loss_names = ['app_gen','correctness_p', 'correctness_r','content_gen','style_gen',
                            'regularization_p', 'regularization_r',
                            'ad_gen','dis_img_gen',
                            'ad_gen_v', 'dis_img_gen_v']

        self.visual_names = ['P_reference','BP_reference', 'P_frame_step','BP_frame_step','img_gen', 'flow_fields', 'masks']        
        self.model_names = ['G','D','D_V']

        self.FloatTensor = torch.cuda.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor

        # define the Animation model
        self.net_G = models.FaceGenerator()
        self.audio2landmark = models.APBNet()
        self.net_D = models.ResDiscriminator()

        input_nc = (opt.frames_D_V-1) * opt.image_nc
        self.net_D_V = models.ResDiscriminator(input_nc=input_nc)
            

        if self.opt.isTrain:
            # define the loss functions
            self.GANloss = external_function.AdversarialLoss(opt.gan_mode).cuda()
            self.L1loss = torch.nn.L1Loss()
            self.L2loss = torch.nn.MSELoss()
            self.Correctness = external_function.PerceptualCorrectness().cuda()
            self.Regularization = external_function.MultiAffineRegularizationLoss(kz_dic=opt.kernel_size).cuda()
            self.Vggloss = external_function.VGGLoss().cuda()

            # define the optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(
                                               filter(lambda p: p.requires_grad, self.net_G.parameters())),
                                               lr=opt.lr, betas=(0.0, 0.999))
            self.optimizers.append(self.optimizer_G)

            self.optimizer_D = torch.optim.Adam(itertools.chain(
                                filter(lambda p: p.requires_grad, self.net_D.parameters()),
                                filter(lambda p: p.requires_grad, self.net_D_V.parameters())),
                                lr=opt.lr*opt.ratio_g2d, betas=(0.0, 0.999))
            self.optimizers.append(self.optimizer_D)
        else:
            self.results_dir_base = self.opt.results_dir
        self.load_networks()

    def load_networks(self,path='weight/'):
        audio2landmark_weight_name = 'audio2landmark.pth'
        if os.path.exists(path+audio2landmark_weight_name):
            weight = torch.load(path+audio2landmark_weight_name,map_location='cuda:0')
            weight = weight['net_G']
            self.audio2landmark.load_state_dict(weight)
            self.audio2landmark.eval()
            self.audio2landmark.cuda()
            print('load audio2landmark')
        else:
            print('file {} does not exists'.format(audio2landmark_weight_name))

        for netname in ['net_D','net_D_V']:
            net = getattr(self,netname)
            if os.path.exists(path+netname+'.pth'):
                print('load {} weight'.format(netname))
                weight = torch.load(path+netname)
                net.load_state_dict(weight)
            else:
                print('weight {} does not find'.format(netname))
            net.cuda()


    def set_input(self, data):
        self.img1 = data['img1'].cuda()
        self.img2 = data['img2']
        self.lab1 = data['lab1']
        self.lab2 = data['lab2']
        self.audio = data['audio']
           

    def write2video(self, name_list):
        images=[]
        for name in name_list:
            images.append(sorted(glob.glob(self.opt.results_dir+'/*_'+name+'.png')))

        image_array=[]
        for i in range(len(images[0])):
            cat_im=None
            for image_list in images:
                im = cv2.imread(image_list[i])
                if cat_im is not None:
                    cat_im = np.concatenate((cat_im, im), axis=1)
                else:
                    cat_im = im
            image_array.append(cat_im) 

        res=''
        for name in name_list:
            res += (name +'_')
        out_name = self.opt.results_dir+'_'+res+'.mp4' 
        print('write video %s'%out_name)  
        height, width, layers = cat_im.shape
        size = (width,height)
        out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
 
        for i in range(len(image_array)):
            out.write(image_array[i])
        out.release()

    def get_current_visuals(self):
        """Return visualization images"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                value = getattr(self, name)
                if 'frame_step' in name:
                    value = value[0]
                    list_value=[]
                    for i in range(value.size(0)):
                        list_value.append(value[i].unsqueeze(0))
                    value=list_value
                if 'flow_field' in name or 'masks' in name:
                    list_value = [item for sub_list in value for item in sub_list]
                    value = list_value

                if isinstance(value, list):
                    # visual multi-scale ouputs
                    for i in range(len(value)):
                        visual_ret[name + str(i)] = self.convert2im(value[i], name)
                    # visual_ret[name] = util.tensor2im(value[-1].data)       
                else:
                    visual_ret[name] =self.convert2im(value, name)         
        return visual_ret 


    def test(self, save_features=False, save_all=False, generate_edge=True):
        """Forward function used in test time"""
        # img_gen, flow_fields, masks = self.net_G(self.input_P1, self.input_BP1, self.input_BP2)
        height, width = self.height, self.width
        image_nc, structure_nc = self.opt.image_nc, self.opt.structure_nc
        n_frames_pre_load = self.opt.n_frames_pre_load_test

        self.BP_frame_step = self.BP_structures.view(-1, n_frames_pre_load, structure_nc, height, width).cuda()
        self.test_generated, self.flow_fields, self.masks, _ = self.net_G(self.BP_frame_step, 
                                                                self.P_reference, 
                                                                self.BP_reference,
                                                                self.P_previous,
                                                                self.BP_previous)
        self.P_previous = self.test_generated[-1] 
        self.BP_previous = self.BP_frame_step[:,-1,... ]   
        
        self.test_generated = torch.cat(self.test_generated, 0)      
        self.save_results(self.test_generated, data_name='vis', data_ext='png')

        if generate_edge:
            value = self.BP_frame_step[:,:,0:1,...][0]
            value = (1-value)*2-1
            self.save_results(value, data_name='edge', data_ext='png')

        if self.change_seq:
            name_list=[] if not generate_edge else ['edge']
            name_list.append('vis')
            print(self.opt.results_dir)
            self.write2video(name_list)


    def optimize_parameters(self):
        self.img2_crop = self.img1[:,0,:,:,:]
        self.lab2_crop = self.lab1[:,0,:,:,:].cuda()
        self.img1_pre = self.img2_crop
        self.lab1_pre = self.lab2_crop
        self.P_previous_recoder = []
        self.img_gen = []
        self.flow_fields = []
        self.masks = []
        for i in range(self.opt.crop_len):
            self.img1_crop = self.img1[:,i,:,:,:]
            self.lab1_crop = self.lab1[:,i,:,:,:].cuda()
            self.audio_crop = self.audio[:,i,:,:]
            self.audio_crop = self.audio_crop[:,None].cuda()
            with torch.no_grad():
                feature = self.audio2landmark(self.audio_crop)

            gen,flowfield,mask = self.net_G(self.lab1_crop,self.img2_crop,self.lab2_crop,self.img1_pre,self.lab1_pre,feature)
            self.img1_pre = gen.detach()
            self.lab1_pre = self.lab1_crop
            self.P_previous_recoder.append(self.img1_pre)
            self.img_gen.append(gen)
            self.flow_fields.append(flowfield)
            self.masks.append(mask)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator"""
        # Real
        D_real = netD(real)
        D_real_loss = self.GANloss(D_real, True, True)
        # fake
        D_fake = netD(fake.detach())
        D_fake_loss = self.GANloss(D_fake, False, True)
        # loss for discriminator
        D_loss = (D_real_loss + D_fake_loss) * 0.5
        # if print_loss:
            # print(D_real_loss)
            # print(D_fake_loss)
        D_loss.backward()

        return D_loss

    def backward_D(self):
        """Calculate the GAN loss for the discriminators"""
        _unfreeze(self.net_D)
        i = np.random.randint(len(self.img_gen))
        fake = self.img_gen[i]
        real = self.img1[:,i,...]
        self.loss_dis_img_gen = self.backward_D_basic(self.net_D, real, fake)

        _unfreeze(self.net_D_V)
        i = np.random.randint(len(self.img_gen)-self.opt.frames_D_V+1)
        # fake = [self.img_gen[i]]
        # real = [self.P_frame_step[:,i,...]]
        fake = []
        real = []        
        for frame in range(self.opt.frames_D_V-1):
            fake.append(self.img_gen[i+frame]-self.img_gen[i+frame+1])
            real.append(self.img1[:,i+frame,...]
                       -self.img1[:,i+frame+1,...])
        fake = torch.cat(fake, dim=1)
        real = torch.cat(real, dim=1)
        self.loss_dis_img_gen_v = self.backward_D_basic(self.net_D_V, real, fake)

    def backward_G(self):
        """Calculate training loss for the generator"""
        # gen_tensor = torch.cat([v.unsqueeze(1) for v in self.img_gen], 1)
        self.style_gen_sum, self.content_gen_sum, self.app_gen_sum=0,0,0

        for i in range(len(self.img_gen)):
            gen = self.img_gen[i]
            gt = self.img1[:,i,...]
            self.app_gen_sum += self.L1loss(gen, gt)

            content_gen, style_gen = self.Vggloss(gen, gt) 
            self.style_gen_sum += style_gen
            self.content_gen_sum += content_gen

        self.loss_style_gen = self.style_gen_sum * self.opt.lambda_style
        self.loss_content_gen = self.content_gen_sum * self.opt.lambda_content            
        self.loss_app_gen = self.app_gen_sum * self.opt.lambda_rec

        self.correctness_p_sum, self.regularization_p_sum=0, 0
        self.correctness_r_sum, self.regularization_r_sum=0, 0

        for i in range(len(self.flow_fields)):
            flow_field_i = self.flow_fields[i]
            flow_p, flow_r=[],[]
            for j in range(0, len(flow_field_i), 2):
                flow_p.append(flow_field_i[j])
                flow_r.append(flow_field_i[j+1])

            correctness_r = self.Correctness(self.img1[:,i,...], self.img2_crop, 
                                                    flow_r, self.opt.attn_layer)
            correctness_p = self.Correctness(self.img1[:,i,...], self.P_previous_recoder[i].detach(), 
                                                    flow_p, self.opt.attn_layer)
            self.correctness_p_sum += correctness_p
            self.correctness_r_sum += correctness_r
            self.regularization_p_sum += self.Regularization(flow_p)
            self.regularization_r_sum += self.Regularization(flow_r)


        self.loss_correctness_p = self.correctness_p_sum * self.opt.lambda_correct     
        self.loss_correctness_r = self.correctness_r_sum * self.opt.lambda_correct   
        self.loss_regularization_p = self.regularization_p_sum * self.opt.lambda_regularization
        self.loss_regularization_r = self.regularization_r_sum * self.opt.lambda_regularization


        # rec loss fake
        _freeze(self.net_D)
        i = np.random.randint(len(self.img_gen))
        fake = self.img_gen[i]
        D_fake = self.net_D(fake)
        self.loss_ad_gen = self.GANloss(D_fake, True, False) * self.opt.lambda_g

        ##########################################################################
        _freeze(self.net_D_V)
        i = np.random.randint(len(self.img_gen)-self.opt.frames_D_V+1)
        # fake = [self.img_gen[i]]
        fake = []
        for frame in range(self.opt.frames_D_V-1):
            fake.append(self.img_gen[i+frame]-self.img_gen[i+frame+1])
        fake = torch.cat(fake, dim=1)
        D_fake = self.net_D_V(fake)
        self.loss_ad_gen_v = self.GANloss(D_fake, True, False) * self.opt.lambda_g
        ##########################################################################
        
        total_loss = 0
        for name in self.loss_names:
            if name != 'dis_img_gen_v' and name != 'dis_img_gen':
                total_loss += getattr(self, "loss_" + name)
        total_loss.backward()

    def run_train(self, dataloader, epoch=None):
        self.epoch += 1
        if epoch:
            self.epoch = epoch
        # self.reset()
        adjust_learning_rate(self.optimizer_G, self.opt.lr, self.epoch, every=self.opt.every)
        adjust_learning_rate(self.optimizer_D, self.opt.lrd, self.epoch, every=self.opt.every)
        for batch_idx, train_data in enumerate(dataloader):
            self.set_input(train_data)
            self.optimize_parameters()
            log_string = 'train\t -> '
            log_string += 'epoch {:>3} '.format(self.epoch)
            log_string += 'batch {:>4} '.format(batch_idx + 1)
            log_string += '|loss_correctness_p {:.5f}'.format(self.correctness_p_sum.item())
            log_string += '|loss_correctness_r {:.5f}'.format(self.correctness_r_sum.item())
            log_string += '|loss_regularization_p_sum {:.5f}'.format(self.regularization_p_sum.item())
            log_string += '|loss_regularization_r_sum {:.5f}'.format(self.regularization_r_sum.item())
            # log_string += '|loss_regularization {:.5f}'.format(self.regularization.item())
            print('\r' + log_string, end='')
            self.show()
        self.save()
        print('\r', end='')
        self.logger.info(log_string)

    def save(self):
        self.net_G.save()
        for netname in ['net_D','net_D_V']:
            net = getattr(self,netname)
            torch.save(net.cpu().state_dict(),'weight/'+netname+'.pth')
            net.cuda()

    def show(self):
        for i in range(len(self.img_gen)):
            tmp = torch.cat([self.img_gen[i],self.img2_crop,self.img1_pre],3)
            tmp = tmp[0].permute(1,2,0).cpu().detach().numpy()
            tmp = (tmp+1)/2*255
            tmp = Image.fromarray(tmp.astype('uint8'))
            tmp.save('tmp{}.jpg'.format(i))
