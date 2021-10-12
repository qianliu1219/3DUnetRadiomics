import os
import torchvision.transforms as transforms
import torchvision.utils as utils
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
from tqdm import tqdm
import shutil
import random
from torch.backends import cudnn
from torch.autograd import Variable
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch.nn.init as nn_init
import math
from easydict import EasyDict as edict
import time
import logging
from tensorboardX import SummaryWriter
import scipy
import scipy.ndimage

root_path = "/home/storage1/qian/TCIA_TCGA-BRCA/"
data_split_path = "/home/storage1/qian/TCIA_TCGA-BRCA/data_split"

batchsize = 1

def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data

class Dataset():
    # data read and preparation
    def __init__(self, root_dir, txt,transform=None):
        self.root_dir = root_dir
        self.txt = txt
        self.volumn_list = [["{0}.npy".format(os.path.join(self.root_dir,"volumn",item))] for item in read_txt(self.txt)]
        self.mask_list = [["{0}_mask.npy".format(os.path.join(self.root_dir,"mask",item))] for item in read_txt(self.txt)]
        self.transform = transform

    def __len__(self):
        return len(self.volumn_list)


    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        volumn_path = self.volumn_list[index][0]
        mask_path = self.mask_list[index][0]
        
        volumn= np.load(volumn_path)
        mask = np.load(mask_path)
        
        volumn = scipy.ndimage.zoom(volumn, (32/volumn.shape[0], 128/volumn.shape[1], 128/volumn.shape[2]), order=1)
        mask = scipy.ndimage.zoom(mask, (32/mask.shape[0], 128/mask.shape[1], 128/mask.shape[2]), order=1)
        
        volumn = volumn[np.newaxis,:,:,:]
        #mask = mask[np.newaxis,:,:,:]
        
        volumn = torch.from_numpy(volumn)
        mask = torch.from_numpy(mask)
        
        return volumn, mask

    
trainset = Dataset(root_dir = root_path,txt='{0}/train.txt'.format(data_split_path))
valset = Dataset(root_dir = root_path,txt='{0}/validation.txt'.format(data_split_path))
testset = Dataset(root_dir = root_path,txt='{0}/test.txt'.format(data_split_path))


def timeit(f):
    """ Decorator to time Any Function """

    def timed(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        seconds = end_time - start_time
        logging.getLogger("Timer").info("   [-] %s : %2.5f sec, which is %2.5f min, which is %2.5f hour" %
                                        (f.__name__, seconds, seconds / 60, seconds / 3600))
        return result

    return timed


def print_cuda_statistics():
    logger = logging.getLogger("Cuda Statistics")
    import sys
    from subprocess import call
    import torch
    logger.info('__Python VERSION:  {}'.format(sys.version))
    logger.info('__pyTorch VERSION:  {}'.format(torch.__version__))
    logger.info('__CUDA VERSION')
    # call(["nvcc", "--version"])
    # logger.info('__CUDNN VERSION:  {}'.format(torch.backends.cudnn.version()))
    # logger.info('__Number CUDA Devices:  {}'.format(torch.cuda.device_count()))
    # logger.info('__Devices')
    call(["nvidia-smi", "--format=csv",
          "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    logger.info('Active CUDA Device: GPU {}'.format(torch.cuda.current_device()))
    logger.info('Available devices  {}'.format(torch.cuda.device_count()))
    logger.info('Current cuda device  {}'.format(torch.cuda.current_device()))


class AverageMeter:
    """
    Class to be an average meter for any average metric like loss, accuracy, etc..
    """

    def __init__(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def val(self):
        return self.avg



# Convolution Operation with weight normalization technique
class WN_Conv3d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, train_scale=False, init_stdv=1.0):
        super(WN_Conv3d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if train_scale:
            self.weight_scale = Parameter(torch.Tensor(out_channels))
        else:
            self.register_buffer('weight_scale', torch.Tensor(out_channels))

        self.train_scale = train_scale
        self.init_mode = False
        self.init_stdv = init_stdv
        self.kernel_size = kernel_size

        self._reset_parameters()

    def _reset_parameters(self):
        self.weight.data.normal_(std=0.05)
        if self.bias is not None:
            self.bias.data.zero_()
        if self.train_scale:
            self.weight_scale.data.fill_(1.)
        else:
            self.weight_scale.fill_(1.)

    def forward(self, input):
        if self.train_scale:
            weight_scale = self.weight_scale
        else:
            weight_scale = Variable(self.weight_scale)
        # normalize weight matrix and linear projection [out x in x h x w x z]
        # for each output dimension, normalize through (in, h, w, z) = (1, 2, 3, 4) dims

        # This is done to ensure padding as "SAME"
        #print(input.shape)
        pad_h = math.ceil((self.kernel_size[0] - input.shape[2] * (1 - self.stride[0]) - self.stride[0]) / 2)
        pad_w = math.ceil((self.kernel_size[1] - input.shape[3] * (1 - self.stride[1]) - self.stride[1]) / 2)
        pad_z = math.ceil((self.kernel_size[2] - input.shape[4] * (1 - self.stride[2]) - self.stride[2]) / 2)
        padding = (pad_h, pad_w, pad_z)

        norm_weight = self.weight * (weight_scale[:,None,None,None,None] / torch.sqrt((self.weight ** 2).sum(4).sum(3).sum(2).sum(1) + 1e-6).reshape([-1, 1, 1, 1, 1])).expand_as(self.weight)
        norm_weight = norm_weight.cuda()
        activation = F.conv3d(input, norm_weight, bias=None,
                              stride=self.stride, padding=padding,
                              dilation=self.dilation, groups=self.groups)

        if self.init_mode == True:
            mean_act = activation.mean(4).mean(3).mean(2).mean(0).squeeze()
            activation = activation - mean_act[None,:,None,None].expand_as(activation)

            inv_stdv = self.init_stdv / torch.sqrt((activation ** 2).mean(4).mean(3).mean(2).mean(0) + 1e-6).squeeze()
            activation = activation * inv_stdv[None,:,None,None,None].expand_as(activation)

            if self.train_scale:
                self.weight_scale.data = self.weight_scale.data * inv_stdv.data
            else:
                self.weight_scale = self.weight_scale * inv_stdv.data
            self.bias.data = - mean_act.data * inv_stdv.data

        else:
            if self.bias is not None:
                activation = activation + self.bias[None,:,None,None,None].expand_as(activation)

        return activation

class WN_ConvTranspose3d(nn.ConvTranspose3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, train_scale=False, init_stdv=1.0):
        super(WN_ConvTranspose3d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias)
        if train_scale:
            self.weight_scale = Parameter(torch.Tensor(out_channels))
        else:
            self.register_buffer('weight_scale', torch.Tensor(out_channels))

        self.train_scale = train_scale
        self.init_mode = False
        self.init_stdv = init_stdv
        self.kernel_size = kernel_size

        self.in_channels = in_channels
        self.out_channels = out_channels

        self._reset_parameters()

    def _reset_parameters(self):
        self.weight.data.normal_(std=0.05)
        if self.bias is not None:
            self.bias.data.zero_()
        if self.train_scale:
            self.weight_scale.data.fill_(1.)
        else:
            self.weight_scale.fill_(1.)

    def forward(self, input, output_size=None):
        if self.train_scale:
            weight_scale = self.weight_scale
        else:
            weight_scale = Variable(self.weight_scale)

        # normalize weight matrix and linear projection [in x out x h x w x z]
        # for each output dimension, normalize through (in, h, w, z)  = (0, 2, 3, 4) dims
        if self.in_channels == self.out_channels:
            norm_weight = self.weight * (weight_scale[None,:,None,None,None] / torch.sqrt((self.weight ** 2).sum(4).sum(3).sum(2).sum(0) + 1e-6).reshape([-1, 1, 1, 1, 1])).expand_as(self.weight)
        else:
            norm_weight = self.weight * (weight_scale[None,:,None,None,None] / torch.sqrt((self.weight ** 2).sum(4).sum(3).sum(2).sum(0) + 1e-6)).expand_as(self.weight)
        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)
        activation = F.conv_transpose3d(input, norm_weight, bias=None,
                                        stride=self.stride, padding=self.padding,
                                        output_padding=output_padding, groups=self.groups)

        if self.init_mode == True:
            mean_act = activation.mean(4).mean(3).mean(2).mean(0).squeeze()
            activation = activation - mean_act[None,:,None,None,None].expand_as(activation)

            inv_stdv = self.init_stdv / torch.sqrt((activation ** 2).mean(4).mean(3).mean(2).mean(0) + 1e-6).squeeze()
            activation = activation * inv_stdv[None,:,None,None,None].expand_as(activation)

            if self.train_scale:
                self.weight_scale.data = self.weight_scale.data * inv_stdv.data
            else:
                self.weight_scale = self.weight_scale * inv_stdv.data
            self.bias.data = - mean_act.data * inv_stdv.data

        else:
            if self.bias is not None:
                activation = activation + self.bias[None,:,None,None,None].expand_as(activation)

        return activation


# 3D-UNet 
class Discriminator(nn.Module):
    def __init__(self, phase):
        super(Discriminator, self).__init__()
        self.phase= phase
        self.input_channel = 1
        self.num_classes = 2
        kernel_size = (3,3,3)
        kernel_size_deconv = (2,2,2)
        stride_deconv = (2,2,2)
        out_channels = 32
        self.lrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout3d(p=0.2, inplace=False)
        self.final_activation = nn.Softmax(dim=1)

        # Defining the convolutional operations
        self.pool = nn.AvgPool3d(2)

        self.encoder0 = WN_Conv3d(self.input_channel, out_channels, kernel_size)
        self.encoder1 = WN_Conv3d(out_channels, out_channels, kernel_size)

        self.encoder2 = WN_Conv3d(out_channels, out_channels*(2), kernel_size)
        self.encoder3 = WN_Conv3d(out_channels*(2), out_channels*(2), kernel_size)

        self.encoder4 = WN_Conv3d(out_channels*(2), out_channels*(2**2), kernel_size)
        self.encoder5 = WN_Conv3d(out_channels*(2**2), out_channels*(2**2), kernel_size)

        self.encoder6 = WN_Conv3d(out_channels*(2**2), out_channels*(2**3), kernel_size)
        self.encoder7 = WN_Conv3d(out_channels*(2**3), out_channels*(2**3), kernel_size)

        self.decoder1 = WN_ConvTranspose3d(out_channels*(2**3), out_channels*(2**3), kernel_size_deconv, stride_deconv)
        #encoder5 + decoder1
        self.encoder8 = WN_Conv3d(out_channels*(2**2) + out_channels*(2**3), out_channels*(2**2), kernel_size)
        self.encoder9 = WN_Conv3d(out_channels*(2**2), out_channels*(2**2), kernel_size)

        self.decoder2 = WN_ConvTranspose3d(out_channels*(2**2), out_channels*(2**2), kernel_size_deconv, stride_deconv)
        #encoder3 + decoder2
        self.encoder10 = WN_Conv3d(out_channels*(2) + out_channels*(2**2), out_channels*(2), kernel_size)
        self.encoder11 = WN_Conv3d(out_channels*(2), out_channels*(2), kernel_size)

        self.decoder3 = WN_ConvTranspose3d(out_channels*(2), out_channels*(2), kernel_size_deconv, stride_deconv)
        #encoder1 + decoder3
        self.encoder12 = WN_Conv3d(out_channels + out_channels*(2), out_channels, kernel_size)
        self.encoder13 = WN_Conv3d(out_channels, out_channels, kernel_size)

        self.final_conv = WN_Conv3d(out_channels, self.num_classes, kernel_size)

    def forward(self, input, get_feature=False, use_dropout=False):
        conv0 = self.lrelu(self.encoder0(input))
        conv1 = self.lrelu(self.encoder1(conv0))
        pool1 = self.pool(conv1)

        conv2 = self.lrelu(self.encoder2(pool1))
        conv3 = self.lrelu(self.encoder3(conv2))
        pool3 = self.pool(conv3)

        conv4 = self.lrelu(self.encoder4(pool3))
        conv5 = self.lrelu(self.encoder5(conv4))
        pool5 = self.pool(conv5)

        conv6 = self.lrelu(self.encoder6(pool5))
        conv7 = self.lrelu(self.encoder7(conv6))

        if use_dropout:
            conv7 = self.dropout(conv7)

        deconv1 = self.decoder1(conv7)
        skip_connection1 = torch.cat((conv5, deconv1), 1)
        conv8 = self.lrelu(self.encoder8(skip_connection1))
        conv9 = self.lrelu(self.encoder9(conv8))

        deconv2 = self.decoder2(conv9)
        skip_connection2 = torch.cat((conv3, deconv2), 1)
        conv10 = self.lrelu(self.encoder10(skip_connection2))
        conv11 = self.lrelu(self.encoder11(conv10))

        deconv3 = self.decoder3(conv11)
        skip_connection3 = torch.cat((conv1, deconv3), 1)
        conv12 = self.lrelu(self.encoder12(skip_connection3))
        conv13 = self.lrelu(self.encoder13(conv12))

        if use_dropout:
            conv13 = self.dropout(conv13)

        final_output = self.final_conv(conv13)

        if not get_feature:
            return final_output, self.final_activation(final_output)
        else:
            return final_output, self.final_activation(final_output), conv6

    
class Supervised_Model():

    def __init__(self,phase,epochs):
        super().__init__()
        self.logger = logging.getLogger("Agent")
        self.phase = phase
        self.net = Discriminator(self.phase) # Segmenation Network
        self.validation_every_epoch = 1
        self.checkpoint_dir = "/home/storage1/qian/TCIA_TCGA-BRCA/checkpoint/"
        self.batchsize = 1
        self.learning_rate = 0.001
        self.beta1, self.beta2 = 0.9, 0.99
        self.epochs = epochs
        if self.phase == 'testing':
            self.testloader = DataLoader(testset, batch_size=self.batchsize, drop_last=False, shuffle=True)
        else:
            self.trainloader = DataLoader(trainset, batch_size=self.batchsize, drop_last=False, shuffle=False)
            self.valloader = DataLoader(valset, batch_size=self.batchsize, drop_last=False, shuffle=False)

        # optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), 
                                          lr=self.learning_rate, 
                                          betas=(self.beta1, self.beta2))

        # counter initialization
        self.current_epoch = 0
        self.best_validation_dice = 0
        self.current_iteration = 0
        self.net = self.net.cuda()
        
        class_weights = torch.tensor([[0.001,0.999]])
        class_weights = torch.FloatTensor(class_weights).cuda()
        self.criterion = nn.CrossEntropyLoss(class_weights)

        self.manual_seed = random.randint(1, 10000)
        self.logger.info ("seed: %d" , self.manual_seed)
        random.seed(self.manual_seed)
        
        self.device = torch.device("cuda")
        
        torch.cuda.manual_seed_all(self.manual_seed)
        
        torch.cuda.set_device(0)
        self.logger.info("Program will run on *****GPU-CUDA***** ")
        print_cuda_statistics()

        self.load_checkpoint(self.phase)

    def load_checkpoint(self, phase):
        try:
            if phase == 'training':
                filename = os.path.join(self.checkpoint_dir, 'checkpoint.pth.tar')
            elif phase == 'testing':
                filename = os.path.join(self.checkpoint_dir, 'model_best.pth.tar')
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint['epoch']
            self.net.load_state_dict(checkpoint['net'])
            self.manual_seed = checkpoint['manual_seed']

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {})\n"
                  .format(self.checkpoint_dir, checkpoint['epoch']))

        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.checkpoint_dir))
            self.logger.info("**First time to train**")


    def save_checkpoint(self, is_best=False):
        file_name="checkpoint.pth.tar"
        state = {
            'epoch': self.current_epoch,
            'net': self.net.state_dict(),
            'manual_seed': self.manual_seed
        }
        torch.save(state, os.path.join(self.checkpoint_dir , file_name))
        if is_best:
            print("SAVING BEST CHECKPOINT !!!")
            shutil.copyfile(self.checkpoint_dir + file_name,
                            self.checkpoint_dir + 'model_best.pth.tar')

    def run(self):
        if self.phase == 'training':
            self.train()
        if self.phase == 'testing':
            self.load_checkpoint(self.phase)
            self.test()


    def train(self):
        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            self.current_iteration = 0
            self.train_one_epoch()
            self.save_checkpoint()
            if(self.current_epoch % self.validation_every_epoch == 0):
                self.validate()

    def train_one_epoch(self):
        # initialize tqdm batch
        tqdm_batch = tqdm(self.trainloader, total=len(self.trainloader), desc="epoch-{}-".format(self.current_epoch))

        self.net.train()
        epoch_loss = AverageMeter()

        for curr_it, (patches, labels) in enumerate(tqdm_batch):
            #y = torch.full((self.batch_size,), self.real_label)
            patches = patches.cuda()
            labels = labels.cuda()

            patches = Variable(patches)
            labels = Variable(labels).long()

            self.net.zero_grad()
            output_logits, output_prob = self.net(patches)
            #print("output_logits:{0}".format(output_logits.shape))
            #print("labels:{0}".format(labels.shape))
            #print(np.unique(labels.cpu().numpy()))
            loss = self.criterion(output_logits, labels)

            loss.backward()
            self.optimizer.step()

            epoch_loss.update(loss.item())
            self.current_iteration += 1
            print("Epoch: {0}, Iteration: {1}/{2}, Loss: {3}".format(self.current_epoch, self.current_iteration,\
                                                                    len(self.trainloader), loss.item()))

        tqdm_batch.close()

        self.logger.info("Training at epoch-" + str(self.current_epoch) + " | " + "Model loss: " + str(epoch_loss.val))

    def validate(self):
        self.net.eval()
        #prediction_image = torch.zeros([10,1,32,128,128])
        for batch_number, (patches, label) in enumerate(self.valloader):
            patches = patches.cuda()
            _, batch_prediction_softmax = self.net(patches)
            prediction_image = torch.argmax(batch_prediction_softmax, dim=1).cpu()
            #prediction_image[batch_number*self.batch_size:(batch_number+1)*self.batch_size,:,:,:] = batch_prediction

            print("Validating.. [{0}/{1}]".format(batch_number, len(self.valloader)))

        #vol_shape_x, vol_shape_y, vol_shape_z = self.volume_shape
        prediction_image = prediction_image.numpy().astype('uint8')
        #val_image_pred = recompose3D_overlap(prediction_image, vol_shape_x, vol_shape_y, vol_shape_z, self.extraction_step[0],
        #                                            self.extraction_step[1],self.extraction_step[2])
        #val_image_pred = val_image_pred.astype('uint8')
        pred2d=np.reshape(prediction_image,(1*32*128*128))
        lab2d=np.reshape(label.numpy().astype('uint8'),(1*32*128*128))
        print("pred2d:{0}".format(pred2d))
        print("lab2d:{0}".format(lab2d))

        classes = list(range(0, 2))
        F1_score = f1_score(lab2d, pred2d, classes, average=None)
        print("Validation Dice Coefficient.... ")
        print("Background:",F1_score[0])
        print("Tumor:",F1_score[1])

        current_validation_dice = F1_score[1]
        if(self.best_validation_dice < current_validation_dice):
            self.best_validation_dice = current_validation_dice
            self.save_checkpoint(is_best = True)

    def test(self):
        self.net.eval()

        #prediction_image = torch.zeros([self.testloader.dataset.patches.shape[0], self.patch_shape[0],\
        #                                self.patch_shape[1], self.patch_shape[2]])
        #whole_vol = self.testloader.dataset.whole_vol
        for batch_number, (patches, label) in enumerate(self.testloader):
            patches = patches.cuda()
            _, batch_prediction_softmax = self.net(patches)
            prediction_image = torch.argmax(batch_prediction_softmax, dim=1).cpu()
    
            print("Testing.. [{0}/{1}]".format(batch_number, len(self.testloader)))

        prediction_image = prediction_image.numpy().astype('uint8')
        pred2d=np.reshape(prediction_image,(1*32*128*128))
        lab2d=np.reshape(label.numpy().astype('uint8'),(1*32*128*128))

        classes = list(range(0, 2))
        F1_score = f1_score(lab2d, pred2d, classes, average=None)
        print("Test Dice Coefficient.... ")
        print("Background:",F1_score[0])
        print("Tumor:",F1_score[1])



    def finalize(self):
        """
        Finalize all the operations of the 2 Main classes of the process the operator and the data loader
        :return:
        """
        self.logger.info("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()
        # self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        # self.summary_writer.close()
        # self.dataloader.finalize()



Supervised_Model('training',1000).run()

Supervised_Model('testing',1000).run()

