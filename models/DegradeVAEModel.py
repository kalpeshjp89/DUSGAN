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


class DegradeVAEModel(BaseModel):
    def __init__(self, opt):
        super(DegradeVAEModel, self).__init__(opt)
        train_opt = opt['train']

        # define networks and load pretrained models
        self.netG = networks.define_G2(opt).to(self.device)  # G1
        if self.is_train:
            self.netD = networks.define_Q(opt).to(self.device)  # G1
            self.netQ = networks.define_Q(opt).to(self.device)
            self.netG.train()
            self.netD.train()
        self.load()  # load G and D if needed

        self.netQ.eval()

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
            self.weight_kl = 1e-2
            self.weight_D = 1e-3

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
            wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_opt['lr_D'], \
                weight_decay=wd_D, betas=(train_opt['beta1_D'], 0.999))
            self.optimizers.append(self.optimizer_D)

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
        self.DR = self.netG(self.var_H)
        self.DR_Encoded = self.netD(self.DR)
        self.LR_Encoded = self.netD(self.var_L)

        n1 = torch.nn.Upsample(scale_factor=0.25)

        l_g_total = 0
        
        Quality1 = self.netQ(self.DR).detach()
        Quality2 = self.netQ(self.var_L).detach()

        Q_loss = 5e-3 * self.cri_pix(Quality1,Quality2)

        l_g_total += Q_loss

        l_g_pix = self.l_pix_w * self.cri_pix(self.DR, n1(self.var_H))
        l_g_total += l_g_pix
        
        l_g_dis = self.weight_D * self.cri_pix(self.LR_Encoded, self.DR_Encoded)
        l_g_total += l_g_dis

        #l_g_tv = 1e-11 * (torch.sum(torch.abs(self.SR[:, :, :, :-1] - self.SR[:, :, :, 1:])) + torch.sum(torch.abs(self.SR[:, :, :-1, :] - self.SR[:, :, 1:, :])))
        #l_g_total +=l_g_tv

        l_g_total.backward()
        self.optimizer_G.step()

        self.optimizer_D.zero_grad()
        log_d_total = 0
        self.LR_Encoded = self.netD(self.var_L)
        
        half_size = int(self.LR_Encoded.shape[1]//2)
        mu = self.LR_Encoded[:,0:half_size]
        logvar = self.LR_Encoded[:,half_size:]
        loss_kl = torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) * (-0.5 * self.weight_kl)
        log_d_total += loss_kl

        log_d_total.backward()
        self.optimizer_D.step()

        # set log
        self.log_dict['l_g_pix'] = l_g_pix.item()
        self.log_dict['l_g_d'] = l_g_dis.item()
        #self.log_dict['l_g_tv'] = l_g_tv.item()
        self.log_dict['l_d_kl'] = loss_kl.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.DR = self.netG(self.var_H)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['SR'] = self.DR.detach()[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.var_L.detach()[0].float().cpu()
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

        if self.is_train:
            # Discriminator
            s, n = self.get_network_description(self.netD)
            if isinstance(self.netD, nn.DataParallel):
                net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
                                                self.netD.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.netD.__class__.__name__)
            logger.info('Network D structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)
        
        load_path_D = self.opt['path']['pretrain_model_D']
        if self.opt['is_train'] and load_path_D is not None:
            logger.info('Loading pretrained model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD)

        load_path_Q = "/media/dl/DL/Kalpesh/Image Quality/experiments/VGGGAP_Kadid_Qualifier/models/latest_G.pth"
        logger.info('Loading pretrained model for Q [{:s}] ...'.format(load_path_Q))
        self.load_network(load_path_Q, self.netQ)

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
        self.save_network(self.netD, 'D', iter_step)
