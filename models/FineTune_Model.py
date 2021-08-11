#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 07:32:18 2020

@author: user1
"""


import os
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import models.networks as networks
from .base_model import BaseModel
from models.modules.loss import GANLoss

logger = logging.getLogger('base')

class GaussianFilter(nn.Module):
    def __init__(self, kernel_size=5, stride=1, padding=4):
        super(GaussianFilter, self).__init__()
        # initialize guassian kernel
        mean = (kernel_size - 1) / 2.0
        variance = (kernel_size / 6.0) ** 2.0
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        # Calculate the 2-dimensional gaussian kernel
        gaussian_kernel = torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(3, 1, 1, 1)

        # create gaussian filter as convolutional layer
        self.gaussian_filter = nn.Conv2d(3, 3, kernel_size, stride=stride, padding=padding, groups=3, bias=False)
        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False

    def forward(self, x):
        return self.gaussian_filter(x)

class FilterLow(nn.Module):
    def __init__(self, recursions=1, kernel_size=5, stride=1, padding=True, include_pad=True, gaussian=True):
        super(FilterLow, self).__init__()
        if padding:
            pad = int((kernel_size - 1) / 2)
        else:
            pad = 0
        if gaussian:
            self.filter = GaussianFilter(kernel_size=kernel_size, stride=stride, padding=pad)
        else:
            self.filter = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=pad, count_include_pad=include_pad)
        self.recursions = recursions

    def forward(self, img):
        for i in range(self.recursions):
            img = self.filter(img)
        return img




class FineTune_Model(BaseModel):
    def __init__(self, opt):
        super(FineTune_Model, self).__init__(opt)
        train_opt = opt['train']

        # define networks and load pretrained models
        self.netG = networks.define_G1(opt).to(self.device)  # G1
        if self.is_train:
            #self.netD = networks.define_D2(opt).to(self.device)  # G1
            self.netTD = networks.define_TrainedD(opt).to(self.device)
            self.netD2 = networks.define_D2(opt).to(self.device)
            #self.vgg = networks.define_F(opt, use_bn=False,Rlu=True).to(self.device)
            self.netQ = networks.define_Q(opt).to(self.device)
            self.netG.train()
            #self.netD.train()
            self.netD2.train()
        self.load()  # load G and D if needed
        self.n1 = torch.nn.Upsample(scale_factor=4,align_corners=True,mode='bicubic').to(self.device) 

        # define losses, optimizer and scheduler
        if self.is_train:
            # G pixel loss
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight']
            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None
            #self.weight_kl = 1e-2
            #self.weight_D = 1e-3
            self.l_gan_w = train_opt['gan_weight']
            self.qa_w = train_opt['QA_weight']
            self.color_filter = FilterLow(recursions=1,kernel_size=5,gaussian=True).to(self.device)

            # G feature loss
            if train_opt['feature_weight'] > 0:
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea_w = train_opt['feature_weight']
            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None
            if self.cri_fea:  # load VGG perceptual loss
                self.netF = networks.define_F(opt, use_bn=False,Rlu=True).to(self.device)   #Rlu=True if feature taken before relu, else false

            self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
            # optimizers
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'], \
                weight_decay=wd_G, betas=(train_opt['beta1_G'], 0.999))
            self.optimizers.append(self.optimizer_G)

            #D
            """wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
                                                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_opt['lr_D'], \
                                                    weight_decay=wd_D, betas=(train_opt['beta1_D'], 0.999))
                                                self.optimizers.append(self.optimizer_D)"""

            wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(), lr=train_opt['lr_D'], \
                weight_decay=wd_D, betas=(train_opt['beta1_D'], 0.999))
            self.optimizers.append(self.optimizer_D2)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, \
                        train_opt['lr_steps'], train_opt['lr_gamma']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()
        # print network
        self.print_network()

    def feed_data(self, data, need_HR=True):
        # LR
        self.var_L = data['LR'].to(self.device)
        if need_HR:  # train or val
            self.var_H = data['HR'].to(self.device)

    def optimize_parameters(self, step):
        # G
        self.optimizer_G.zero_grad()
        self.SR = self.netG(self.var_L)
        
        #SR_low = self.color_filter(self.SR)
        #HR_low = self.color_filter(self.var_H).detach()
        
        self.SR_Encoded = self.netTD(self.SR).detach()
        self.SR_Encoded2 = self.netD2(self.SR - self.color_filter(self.SR))
        #self.HR_Encoded = self.netTD(self.var_H)
        #self.HR_Encoded2 = self.netD2(self.var_H - self.color_filter(self.var_H))
        #self.SR_Encoded2 = self.netD2(self.vgg(self.SR))
        Quality_loss = self.qa_w * torch.exp(-0.5*(torch.mean(self.netQ(self.SR).detach())-5))
        #Quality_loss = self.qa_w * (5-torch.mean(self.netQ(self.SR).detach()))

        #n1 = torch.nn.Upsample(scale_factor=4,align_corners=True,mode='bicubic')

        l_g_total = 0
        
        #l_g_pix = self.l_pix_w * self.cri_pix(self.color_filter(self.SR), self.color_filter(n1(self.var_L)))
        l_g_pix = self.l_pix_w * self.cri_pix(self.SR, self.n1(self.var_L))
        l_g_total += l_g_pix
        
        l_g_dis = self.l_gan_w * self.cri_gan(self.SR_Encoded, True) #simple gn
        #l_g_dis = self.l_gan_w * (self.cri_gan(self.SR_Encoded - torch.mean(self.HR_Encoded), True) + self.cri_gan(self.HR_Encoded - torch.mean(self.SR_Encoded), False))/ 2 #rgn
        l_g_total += l_g_dis
        l_g_dis2 = self.l_gan_w * self.cri_gan(self.SR_Encoded2, True) #simple gn
        #l_g_dis2 = self.l_gan_w * (self.cri_gan(self.SR_Encoded2 - torch.mean(self.HR_Encoded2), True) + self.cri_gan(self.HR_Encoded2 - torch.mean(self.SR_Encoded2), False))/ 2 
        l_g_total += l_g_dis2

        #sr_g= (torch.pow((self.SR[:, :, 1:, :-1] - self.SR[:, :, 1:, 1:]),2) + torch.pow((self.SR[:, :, :-1, 1:] - self.SR[:, :, 1:, 1:]),2)) / (0.8)**2
        #l_grad = 2e-7 * torch.sum(torch.min(sr_g,torch.ones_like(sr_g).to(self.device)))
        #l_g_total +=l_g_tv

        l_g_total += Quality_loss
        #l_g_total += l_grad

        l_g_total.backward()
        self.optimizer_G.step()

        """self.optimizer_D.zero_grad()
                                log_d_total = 0
                                self.SR = self.netG(self.var_L)
                                
                                #SR_low = self.color_filter(self.SR)
                                #HR_low = self.color_filter(self.var_H)
                                self.HR_Encoded = self.netD(self.var_H)
                                self.SR_Encoded = self.netD(self.SR)
                                
                                g1 = self.l_gan_w * self.cri_gan(self.HR_Encoded - torch.mean(self.SR_Encoded), True)
                                g2 = self.l_gan_w * self.cri_gan(self.SR_Encoded - torch.mean(self.HR_Encoded), False)
                                log_d_total += (g1 + g2)*0.5
                        
                        
                                log_d_total.backward()
                                self.optimizer_D.step()"""

        self.optimizer_D2.zero_grad()
        log_d2_total = 0
        self.SR = self.netG(self.var_L)
        
        #SR_low = self.color_filter(self.SR)
        #HR_low = self.color_filter(self.var_H)
        self.HR_Encoded2 = self.netD2(self.var_H - self.color_filter(self.var_H))
        self.SR_Encoded2 = self.netD2(self.SR - self.color_filter(self.SR))
        #self.HR_Encoded2 = self.netD2(self.vgg(self.var_H))
        #self.SR_Encoded2 = self.netD2(self.vgg(self.SR))
        
        g1 = self.l_gan_w * self.cri_gan(self.HR_Encoded2 - torch.mean(self.SR_Encoded2), True)
        g2 = self.l_gan_w * self.cri_gan(self.SR_Encoded2 - torch.mean(self.HR_Encoded2), False)
        log_d2_total += (g1 + g2)*0.5


        log_d2_total.backward()
        self.optimizer_D2.step()

        # set log
        self.log_dict['l_g_pix'] = l_g_pix.item()
        self.log_dict['l_g_d'] = l_g_dis.item()
        self.log_dict['l_g_d2'] = l_g_dis2.item()
        #self.log_dict['l_grad'] = l_grad.item()
        self.log_dict['l_g_qa'] = Quality_loss.item()
        #self.log_dict['d_total'] = log_d_total.item()
        self.log_dict['d2_total'] = log_d2_total.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.SR = self.netG(self.var_L)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['SR'] = self.SR.detach()[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.var_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

        """if self.is_train:
                                    # Discriminator
                                    s, n = self.get_network_description(self.netD)
                                    if isinstance(self.netD, nn.DataParallel):
                                        net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
                                                                        self.netD.module.__class__.__name__)
                                    else:
                                        net_struc_str = '{}'.format(self.netD.__class__.__name__)
                                    logger.info('Network D structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
                                    logger.info(s)"""

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)
        
        load_path_TD = '/home/user1/Documents/Kalpesh/NTIRE2_Code/129000_D.pth'
        if self.opt['is_train']:
            logger.info('Loading pretrained model for TD [{:s}] ...'.format(load_path_TD))
            self.load_network(load_path_TD, self.netTD)
        load_path_D2 = self.opt['path']['pretrain_model_D2']
        if self.opt['is_train'] and load_path_D2 is not None:
            logger.info('Loading pretrained model for D2 [{:s}] ...'.format(load_path_D2))
            self.load_network(load_path_D2, self.netD2)
        if self.opt['is_train']:
            load_path_Q = "/home/user1/Documents/Kalpesh/NTIRE2_Code/latest_G.pth"
            logger.info('Loading pretrained model for Q [{:s}] ...'.format(load_path_Q))
            self.load_network(load_path_Q, self.netQ)
    

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
        #self.save_network(self.netD, 'D', iter_step)
        self.save_network(self.netD2, 'D2', iter_step)
    
    
    