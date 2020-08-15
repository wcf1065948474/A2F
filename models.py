import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from block_extractor.block_extractor  import BlockExtractor
from local_attn_reshape.local_attn_reshape   import LocalAttnReshape

class EncoderBlock(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(EncoderBlock, self).__init__()

        kwargs_down = {'kernel_size': 4, 'stride': 2, 'padding': 1}
        kwargs_fine = {'kernel_size': 3, 'stride': 1, 'padding': 1}

        self.conv1 = nn.Conv2d(input_nc,  output_nc, **kwargs_down)
        self.conv2 = nn.Conv2d(output_nc, output_nc, **kwargs_fine)

        self.norm1 = nn.InstanceNorm2d(input_nc)
        self.norm2 = nn.InstanceNorm2d(output_nc)

    def forward(self, x):
        out = self.norm1(x)
        out = F.leaky_relu(out,0.1)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.leaky_relu(out,0.1)
        out = self.conv2(out)
        return out


class ResBlockDecoderWithAudioFeature(nn.Module):
    def __init__(self, input_nc, output_nc, hidden_nc, shape):
        super(ResBlockDecoderWithAudioFeature, self).__init__()
        self.shape = shape
        self.main_up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.bypass_up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.main_conv = nn.Conv2d(input_nc, output_nc, kernel_size=1, stride=1)
        self.bypass_conv1 = nn.Conv2d(hidden_nc, hidden_nc, kernel_size=3, stride=1, padding=1)
        self.bypass_conv2 = nn.Conv2d(hidden_nc, output_nc, kernel_size=3, stride=1, padding=1)
        self.linear = nn.Linear(512,shape[0]*shape[1]*shape[2])

        self.norm1 = nn.InstanceNorm2d(hidden_nc)
        self.norm2 = nn.InstanceNorm2d(hidden_nc)
        self.norm3 = nn.InstanceNorm2d(input_nc)

    def forward(self, x, feature):
        bypass_out = self.linear(feature)
        bypass_out = bypass_out.view(-1,self.shape[0],self.shape[1],self.shape[2])
        bypass_out = torch.cat((x,bypass_out),1)
        bypass_out = self.norm1(bypass_out)
        bypass_out = F.leaky_relu(bypass_out,0.1)
        bypass_out = self.bypass_up(bypass_out)
        bypass_out = self.bypass_conv1(bypass_out)
        bypass_out = self.norm2(bypass_out)
        bypass_out = F.leaky_relu(bypass_out,0.1)
        bypass_out = self.bypass_conv2(bypass_out)

        x = self.norm3(x)
        x = F.leaky_relu(x,0.1)
        main_out = self.main_up(x)
        main_out = self.main_conv(main_out)

        out = main_out + bypass_out
        return out



# class ResBlockDecoder(nn.Module):
#     def __init__(self, input_nc, output_nc, hidden_nc=None):
#         super(ResBlockDecoder, self).__init__()

#         hidden_nc = input_nc if hidden_nc is None else hidden_nc

#         conv1 = nn.Conv2d(input_nc, hidden_nc, kernel_size=3, stride=1, padding=1)
#         conv2 = nn.ConvTranspose2d(hidden_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1)
#         bypass = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1)

#         self.model = nn.Sequential(nn.InstanceNorm2d(input_nc), nn.LeakyReLU(0.1), conv1, nn.InstanceNorm2d(hidden_nc), nn.LeakyReLU(0.1), conv2,)

#         self.shortcut = nn.Sequential(bypass)

#     def forward(self, x):
#         out = self.model(x) + self.shortcut(x)
#         return out


class ResBlockDecoder(nn.Module):
    """
    Define a decoder block
    """
    def __init__(self, input_nc, output_nc, hidden_nc=None):
        super(ResBlockDecoder, self).__init__()

        hidden_nc = input_nc if hidden_nc is None else hidden_nc

        conv1 = nn.Conv2d(input_nc, hidden_nc, kernel_size=3, stride=1, padding=1)
        # conv2 = spectral_norm(nn.ConvTranspose2d(hidden_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1), use_spect)
        # bypass = spectral_norm(nn.ConvTranspose2d(input_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1), use_spect)
        conv2_up = nn.UpsamplingBilinear2d(scale_factor=2)
        bypass_up = nn.UpsamplingBilinear2d(scale_factor=2)
        conv2 = nn.Conv2d(hidden_nc, output_nc, kernel_size=3, stride=1, padding=1)
        bypass = nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1)

        self.model = nn.Sequential(nn.InstanceNorm2d(input_nc), nn.LeakyReLU(0.1), conv1, nn.InstanceNorm2d(hidden_nc), nn.LeakyReLU(0.1),conv2_up, conv2,)

        self.shortcut = nn.Sequential(bypass_up,bypass)

    def forward(self, x):
        out = self.model(x) + self.shortcut(x)
        return out


class ResBlockEncoder(nn.Module):
    def __init__(self, input_nc, output_nc, hidden_nc=None):
        super(ResBlockEncoder, self).__init__()
        norm_layer = None
        hidden_nc = input_nc if hidden_nc is None else hidden_nc

        conv1 = nn.Conv2d(input_nc, hidden_nc, kernel_size=3, stride=1, padding=1)
        conv2 = nn.Conv2d(hidden_nc, output_nc, kernel_size=4, stride=2, padding=1)
        bypass = nn.Conv2d(input_nc, output_nc, kernel_size=1, stride=1, padding=0)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nn.LeakyReLU(0.1), conv1, nn.LeakyReLU(0.1), conv2,)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nn.LeakyReLU(0.1), conv1, 
                                       norm_layer(hidden_nc), nn.LeakyReLU(0.1), conv2,)
        self.shortcut = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2),bypass)

    def forward(self, x):
        out = self.model(x) + self.shortcut(x)
        return out 


class ResBlock(nn.Module):
    def __init__(self, input_nc, output_nc=None, hidden_nc=None):
        super(ResBlock, self).__init__()

        hidden_nc = input_nc if hidden_nc is None else hidden_nc
        output_nc = input_nc if output_nc is None else output_nc
        self.learnable_shortcut = True if input_nc != output_nc else False

        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}

        conv1 = nn.Conv2d(input_nc, output_nc, **kwargs)
        conv2 = nn.Conv2d(input_nc, output_nc, **kwargs)


        self.model = nn.Sequential(nn.InstanceNorm2d(input_nc), nn.LeakyReLU(0.1), conv1, 
                                    nn.InstanceNorm2d(hidden_nc), nn.LeakyReLU(0.1), conv2,)

        if self.learnable_shortcut:
            bypass = nn.Conv2d(input_nc, output_nc, **kwargs_short)
            self.shortcut = nn.Sequential(bypass,)


    def forward(self, x):
        if self.learnable_shortcut:
            out = self.model(x) + self.shortcut(x)
        else:
            out = self.model(x) + x
        return out

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, input_nc, output_nc=None, hidden_nc=None):
        super(ResBlocks, self).__init__()
        hidden_nc = input_nc if hidden_nc is None else hidden_nc
        output_nc = input_nc if output_nc is None else output_nc

        self.model=[]
        if num_blocks == 1:
            self.model += [ResBlock(input_nc, output_nc, hidden_nc)]

        else:
            self.model += [ResBlock(input_nc, hidden_nc, hidden_nc)]
            for i in range(num_blocks-2):
                self.model += [ResBlock(hidden_nc, hidden_nc, hidden_nc)]
            self.model += [ResBlock(hidden_nc, output_nc, hidden_nc)]

        self.model = nn.Sequential(*self.model)

    def forward(self, inputs):
        return self.model(inputs)


class Output(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size = 3):
        super(Output, self).__init__()

        kwargs = {'kernel_size': kernel_size, 'padding':0, 'bias': True}

        self.conv1 = nn.Conv2d(input_nc, output_nc, **kwargs)

        self.model = nn.Sequential(nn.InstanceNorm2d(input_nc), nn.LeakyReLU(0.1), nn.ReflectionPad2d(int(kernel_size / 2)), self.conv1, nn.Tanh())

    def forward(self, x):
        out = self.model(x)

        return out


class ExtractorAttn(nn.Module):
    def __init__(self, feature_nc, kernel_size=4, nonlinearity=nn.LeakyReLU(), softmax=None):
        super(ExtractorAttn, self).__init__()
        self.kernel_size=kernel_size
        hidden_nc = 128
        softmax = nonlinearity if softmax is None else nn.Softmax(dim=1)

        self.extractor = BlockExtractor(kernel_size=kernel_size)
        self.reshape = LocalAttnReshape()
        self.fully_connect_layer = nn.Sequential(
                nn.Conv2d(2*feature_nc, hidden_nc, kernel_size=kernel_size, stride=kernel_size, padding=0),
                nonlinearity,
                nn.Conv2d(hidden_nc, kernel_size*kernel_size, kernel_size=1, stride=1, padding=0),
                softmax,)        
    def forward(self, source, target, flow_field):
        block_source = self.extractor(source, flow_field)
        block_target = self.extractor(target, torch.zeros_like(flow_field))
        attn_param = self.fully_connect_layer(torch.cat((block_target, block_source), 1))
        attn_param = self.reshape(attn_param, self.kernel_size)
        result = torch.nn.functional.avg_pool2d(attn_param*block_source, self.kernel_size, self.kernel_size)
        return result

    def hook_attn_param(self, source, target, flow_field):
        block_source = self.extractor(source, flow_field)
        block_target = self.extractor(target, torch.zeros_like(flow_field))
        attn_param_ = self.fully_connect_layer(torch.cat((block_target, block_source), 1))
        attn_param = self.reshape(attn_param_, self.kernel_size)
        result = torch.nn.functional.avg_pool2d(attn_param*block_source, self.kernel_size, self.kernel_size)
        return attn_param_, result


class FlowNet(nn.Module):
    def __init__(self, ngf=32, img_f=256):
        super(FlowNet,self).__init__()
        input_nc = 3+3+3+3+3
        encoder_layer = 5
        self.encoder_layer = encoder_layer
        decoder_layer = encoder_layer - 2
        self.decoder_layer = decoder_layer
        attn_layer = [2,3]
        self.attn_layer = attn_layer
        mult = 1


        self.block0 = EncoderBlock(input_nc, ngf)
        for i in range(encoder_layer-1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f//ngf)
            block = EncoderBlock(ngf*mult_prev, ngf*mult)
            setattr(self, 'encoder' + str(i), block)
        self.decoder0 = ResBlockDecoderWithAudioFeature(256,256,256+16,(16,8,8))
        self.decoder1 = ResBlockDecoderWithAudioFeature(256,128,256+16,(16,16,16))
        self.decoder2 = ResBlockDecoderWithAudioFeature(128,64,128+16,(16,32,32))
        self.output1 = nn.Conv2d(128, 4, kernel_size=3,stride=1,padding=1,bias=True)
        self.mask1 = nn.Sequential(nn.Conv2d(128, 2, kernel_size=3,stride=1,padding=1,bias=True),nn.Sigmoid())
        self.output2 = nn.Conv2d(64, 4, kernel_size=3,stride=1,padding=1,bias=True)
        self.mask2 = nn.Sequential(nn.Conv2d(64, 2, kernel_size=3,stride=1,padding=1,bias=True),nn.Sigmoid())

    def forward(self,BP, P_previous, BP_previous, P_reference, BP_reference,feature):
        flow_fields=[]
        masks=[]
        img = torch.cat((BP, P_previous, BP_previous, P_reference, BP_reference), 1)
        out = self.block0(img)
        # result=[out]
        for i in range(self.encoder_layer-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            # result.append(out) 
            
        for i in range(self.decoder_layer):
            model = getattr(self, 'decoder' + str(i))
            out = model(out,feature)

            # model = getattr(self, 'jump' + str(i))
            # jump = model(result[self.encoder_layer-i-2])
            # out = out+jump

            if self.encoder_layer-i-1 in self.attn_layer:
                flow_field, mask = self.attn_output(out, i)
                flow_field_p, flow_field_r = torch.split(flow_field, 2, dim=1)
                mask_p, mask_r = torch.split(mask, 1, dim=1)
                flow_fields.append(flow_field_p)
                flow_fields.append(flow_field_r)
                masks.append(mask_p)
                masks.append(mask_r)
        return flow_fields, masks

    def attn_output(self, out, i):
        model = getattr(self, 'output' + str(i))
        flow = model(out)
        model = getattr(self, 'mask' + str(i))
        mask = model(out)
        return flow, mask  



class APBNet(nn.Module):

    def __init__(self, num_landmark=212):
        super(APBNet, self).__init__()
        self.num_landmark = num_landmark
        self.lstm = nn.LSTM(512, 256, 2, batch_first=True, bidirectional=True)
        # audio
        self.audio1 = nn.Sequential(
            nn.Conv2d(1, 72, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)), nn.ReLU(),
            nn.Conv2d(72, 108, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)), nn.ReLU(),
            nn.Conv2d(108, 162, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)), nn.ReLU(),
            nn.Conv2d(162, 243, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)), nn.ReLU(),
            nn.Conv2d(243, 256, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)), nn.ReLU(),
        )
        self.audio2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(4, 1), stride=(4, 1)), nn.ReLU()
        )
        self.trans_audio = nn.Sequential(nn.Linear(256 * 2, 256))
        self.latent2poseeye = nn.Sequential(
            nn.Linear(64,64), nn.ReLU(),
            nn.Linear(64,64), nn.ReLU(),
            nn.Linear(64,5)
        )
        # pose
        self.trans_pose_eye = nn.Sequential(
            nn.Linear(5, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 128)
        )
        # eye
        # self.trans_eye = nn.Sequential(
        #     nn.Linear(2, 64), nn.ReLU(),
        #     nn.Linear(64, 64), nn.ReLU(),
        #     nn.Linear(64, 64)
        # )
        # cat
        self.trans_cat = nn.Sequential(
            nn.Linear(256 + 64 * 2, 240), nn.ReLU(),
            nn.Linear(240, self.num_landmark)
        )

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, audio, z):
        b,c,w,h = audio.size()
        audiolist = []
        for i in range(audio.size(1)):
            a = audio[:,i,:,:]
            a = self.audio1(a[:,None])
            audiolist.append(self.audio2(a))
        audio = torch.cat(audiolist,2)
        audio = audio.permute((0,2,1,3))
        audio = audio.reshape(b,c,-1)
        self.lstm.flatten_parameters()
        audio, _ = self.lstm(audio)
        audio = audio.reshape(-1,512)
        pose_eye = self.latent2poseeye(z)
        x_a = self.trans_audio(audio)
        x_p_e = self.trans_pose_eye(pose_eye)
        # x_e = self.trans_eye(pose_eye[:,2:])
        x_cat = torch.cat([x_a, x_p_e], dim=1)
        output = self.trans_cat(x_cat)
        return output, audio, pose_eye


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        layers1 = [nn.Linear(106 * 2, 512), nn.LeakyReLU(0.2, True),
                  nn.Linear(512, 512), nn.LeakyReLU(0.2, True),
                  nn.Linear(512, 512), nn.LeakyReLU(0.2, True),
                  nn.Linear(512, 512), nn.LeakyReLU(0.2, True),
                  nn.Linear(512, 64)]

        layers2 = [nn.Linear(106 * 2, 512), nn.LeakyReLU(0.2, True),
                   nn.Linear(512, 512), nn.LeakyReLU(0.2, True),
                   nn.Linear(512, 512), nn.LeakyReLU(0.2, True),
                   nn.Linear(512, 512), nn.LeakyReLU(0.2, True),
                   nn.Linear(512, 64)]

        layers3 = [nn.Linear(128, 128), nn.LeakyReLU(0.2, True),
                   nn.Linear(128, 32), nn.LeakyReLU(0.2, True),
                   nn.Linear(32, 1)]

        self.layers1 = nn.Sequential(*layers1)
        self.layers2 = nn.Sequential(*layers2)
        self.layers3 = nn.Sequential(*layers3)

    def forward(self, input1, input2):
        x1 = self.layers1(input1)
        x2 = self.layers2(input2)
        x_cat = torch.cat([x1, x2], dim=1)
        out = self.layers3(x_cat)
        return out



class Discriminator_poseeye(nn.Module):
    def __init__(self):
        super(Discriminator_poseeye, self).__init__()
        layers1 = [nn.Linear(5, 64), nn.LeakyReLU(0.2, True),
                  nn.Linear(64, 128), nn.LeakyReLU(0.2, True),
                  nn.Linear(128, 256), nn.LeakyReLU(0.2, True),
                  nn.Linear(256, 128), nn.LeakyReLU(0.2, True),
                  nn.Linear(128, 64)]

        layers2 = [nn.Linear(5, 64), nn.LeakyReLU(0.2, True),
                   nn.Linear(64, 128), nn.LeakyReLU(0.2, True),
                   nn.Linear(128, 256), nn.LeakyReLU(0.2, True),
                   nn.Linear(256, 128), nn.LeakyReLU(0.2, True),
                   nn.Linear(128, 64)]

        layers3 = [nn.Linear(128, 128), nn.LeakyReLU(0.2, True),
                   nn.Linear(128, 32), nn.LeakyReLU(0.2, True),
                   nn.Linear(32, 1)]

        self.layers1 = nn.Sequential(*layers1)
        self.layers2 = nn.Sequential(*layers2)
        self.layers3 = nn.Sequential(*layers3)

    def forward(self, input1, input2):
        x1 = self.layers1(input1)
        x2 = self.layers2(input2)
        x_cat = torch.cat([x1, x2], dim=1)
        out = self.layers3(x_cat)
        return out

class ResDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=32, img_f=128, layers=4, norm='none'):
        super(ResDiscriminator, self).__init__()

        self.layers = layers

        self.nonlinearity = nn.LeakyReLU(0.1)

        # encoder part
        self.block0 = ResBlockEncoder(input_nc, ndf, ndf)

        mult = 1
        for i in range(layers - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f//ndf)
            block = ResBlockEncoder(ndf*mult_prev, ndf*mult, ndf*mult_prev)
            setattr(self, 'encoder' + str(i), block)
        self.conv = nn.Conv2d(ndf*mult, 1, 1)

    def forward(self, x):
        out = self.block0(x)
        for i in range(self.layers - 1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
        out = self.conv(self.nonlinearity(out))
        return out


class PoseSourceNet(nn.Module):
    def __init__(self, input_nc=3, ngf=64, img_f=1024, layers=6):  
        super(PoseSourceNet, self).__init__()
        self.layers = layers

        # encoder part CONV_BLOCKS
        self.block0 = EncoderBlock(input_nc, ngf)
        mult = 1
        for i in range(layers-1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f//ngf)
            block = EncoderBlock(ngf*mult_prev, ngf*mult)
            setattr(self, 'encoder' + str(i), block)        


    def forward(self, source):
        feature_list=[source]
        out = self.block0(source)
        feature_list.append(out)
        for i in range(self.layers-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out) 
            feature_list.append(out)

        feature_list = list(reversed(feature_list))
        return feature_list



class FaceGenerator(nn.Module):
    def __init__(self,  image_nc=3, structure_nc=3, output_nc=3, ngf=64, img_f=512, layers=3, num_blocks=2, 
                 attn_layer=[2,3], extractor_kz={'2': 5, '3': 3}):  
        super(FaceGenerator, self).__init__()
        self.source_previous = PoseSourceNet(image_nc, ngf, img_f, layers)
        self.source_reference = PoseSourceNet(image_nc, ngf, img_f, layers)        
        self.target = FaceTargetNet(image_nc, structure_nc, output_nc, ngf, img_f, layers, num_blocks, attn_layer, extractor_kz)
        self.flow_net = FlowNet()

        self.load_networks()

    def load_networks(self,path='weight/',netnames=['source_previous','source_reference','target','flow_net']):
        for netname in netnames:
            net = getattr(self,netname)
            if os.path.exists(path+netname+'.pth'):
                print('load {} weight'.format(netname))
                weight = torch.load(path+netname)
                net.load_state_dict(weight)
            else:
                print('weight {} does not find'.format(netname))
            net.cuda()

    def save(self):
        for netname in ['source_previous','source_reference','target','flow_net']:
            net = getattr(self,netname)
            torch.save(net.cpu().state_dict(),'weight/'+netname+'.pth')
            net.cuda()

    def forward(self, BP, P_reference, BP_reference, P_previous, BP_previous, feature):
        previous_feature_list = self.source_previous(P_previous)
        reference_feature_list = self.source_reference(P_reference)

        flow_fields, masks = self.flow_net(BP, P_previous, BP_previous, P_reference, BP_reference,feature)
        image_gen = self.target(BP, previous_feature_list, reference_feature_list, flow_fields, masks)

        return image_gen, flow_fields, masks


class FaceTargetNet(nn.Module):

    def __init__(self, image_nc=3, structure_nc=3, output_nc=3, ngf=64, img_f=1024, layers=6, num_blocks=2, 
                attn_layer=[1,2], extractor_kz={'1':5,'2':5}):  
        super(FaceTargetNet, self).__init__()

        self.layers = layers
        self.attn_layer = attn_layer

        self.block0 = EncoderBlock(structure_nc, ngf)
        mult = 1
        for i in range(layers-1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f//ngf)
            block = EncoderBlock(ngf*mult_prev, ngf*mult)
            setattr(self, 'encoder' + str(i), block)         

        # decoder part
        mult = min(2 ** (layers-1), img_f//ngf)
        for i in range(layers):
            mult_prev = mult
            mult = min(2 ** (layers-i-2), img_f//ngf) if i != layers-1 else 1
            if num_blocks == 1:
                up = nn.Sequential(ResBlockDecoder(ngf*mult_prev, ngf*mult, None))
            else:
                up = nn.Sequential(ResBlocks(num_blocks-1, ngf*mult_prev, None, None),
                                   ResBlockDecoder(ngf*mult_prev, ngf*mult, None))
            setattr(self, 'decoder' + str(i), up)

            if layers-i in attn_layer:
                attn = ExtractorAttn(ngf*mult_prev, extractor_kz[str(layers-i)], nn.LeakyReLU(0.1), softmax=True)
                setattr(self, 'attn_p' + str(i), attn)

                attn = ExtractorAttn(ngf*mult_prev, extractor_kz[str(layers-i)], nn.LeakyReLU(0.1), softmax=True)
                setattr(self, 'attn_r' + str(i), attn)

        self.outconv = Output(ngf, output_nc, 3)


    def forward(self, BP, previous_feature_list, reference_feature_list, flow_fields, masks):
        out = self.block0(BP)
        for i in range(self.layers-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out) 

        counter=0
        for i in range(self.layers):
            if self.layers-i in self.attn_layer:
                model_p = getattr(self, 'attn_p' + str(i))
                model_r = getattr(self, 'attn_r' + str(i))

                out_attn_p = model_p(previous_feature_list[i], out, flow_fields[2*counter])        
                out_attn_r = model_r(reference_feature_list[i], out, flow_fields[2*counter+1])        
                out_p = out*(1-masks[2*counter])   + out_attn_p*masks[2*counter]
                out_r = out*(1-masks[2*counter+1]) + out_attn_r*masks[2*counter+1]
                out = out_p + out_r 
                counter += 1

            model = getattr(self, 'decoder' + str(i))
            out = model(out)

        out_image = self.outconv(out)
        return out_image
