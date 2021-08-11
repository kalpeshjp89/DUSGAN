from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
####################
# Basic blocks
####################


def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act_type == 'sigm':
        layer = nn.Sigmoid()
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


def norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


class ConcatBlock(nn.Module):
    # Concat the output of a submodule to its input
    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), dim=1)
        return output

    def __repr__(self):
        tmpstr = 'Identity .. \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


class ShortcutBlock(nn.Module):
    #Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True, \
               pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
    '''
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    '''
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)

def trans_conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True, \
               pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
    '''
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    '''
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    #padding = 1
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.ConvTranspose2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups,output_padding=1)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)

####################
# Useful blocks
####################

class high_block(nn.Module):
    def __init__(self, in_nc, kernel_size=3,norm_type='batch', act_type='leakyrelu'):
        super(high_block, self).__init__()

        self.conv0 = conv_block(in_nc, in_nc, kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
        self.conv1 = conv_block(in_nc, in_nc, kernel_size=kernel_size, norm_type=norm_type,act_type=act_type)
        self.conv2 = conv_block(in_nc, in_nc, kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
        #self.gap = nn.AdaptiveAvgPool2d((1,1))
        #self.conv3 = conv_block(in_nc, 16, kernel_size=1, norm_type=None, act_type='prelu')
        #self.conv4 = conv_block(16, in_nc, kernel_size=1, norm_type=None, act_type='sigm')

    def forward(self,x):
        x1 = self.conv2(self.conv1(self.conv0(x)))
        #m = self.conv4(self.conv3(self.gap(x1)))
        #x2 = x1.mul(m)
        return x1.mul(0.2) + x

class base_block(nn.Module):
    def __init__(self, in_nc, kernel_size=3,norm_type='batch', act_type='leakyrelu'):
        super(base_block, self).__init__()

        self.h0 = high_block(in_nc, kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
        self.h1 = high_block(in_nc, kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
        self.h2 = high_block(in_nc, kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
        self.h3 = high_block(in_nc, kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
        self.h4 = high_block(in_nc, kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
        self.h5 = high_block(in_nc, kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
        self.r1 = conv_block(in_nc,int(0.5*in_nc),kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
        self.r2 = conv_block(in_nc,int(0.5*in_nc),kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
        self.r3 = conv_block(in_nc,int(0.5*in_nc),kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
        self.r4 = conv_block(in_nc,int(0.5*in_nc),kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
        self.r5 = conv_block(in_nc,int(0.5*in_nc),kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
        self.r6 = conv_block(in_nc,int(0.5*in_nc),kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
        #self.gap = nn.AdaptiveAvgPool2d((3,3))
        #self.c0 = conv_block(in_nc*3*3,in_nc*3*3*in_nc,kernel_size=1,norm_type=None,act_type='prelu') #(batch,n*n*3*3,1,1)
        #self.n = in_nc

    def forward(self,x):
        x1 = self.h1(self.h0(x))
        x2 = torch.cat((self.r1(x),self.r2(x1)),1)
        x3 = self.h3(self.h2(x2))
        x4 = torch.cat((self.r3(x2),self.r4(x3)),1)
        x5 = self.h5(self.h4(x4))
        x6 = torch.cat((self.r5(x4),self.r6(x5)),1)
        
        #xc = x1+x2
        #dk = self.gap(x)  #(batch,channel,3,3)
        #dk = torch.reshape(dk,(dk.size(0),self.n*3*3,1,1))
        #dk = self.c0(dk)  #(batch,n*n*3*3)
        #dk = torch.reshape(dk,(dk.size(0),self.n,self.n,3,3))
        #outputs = []
        #for idx in range(dk.size(0)):
        #    input = xc[idx:idx+1]
        #    weight = dk[idx]
        #    output = F.conv2d(input, weight, stride=1, padding=1)
        #    outputs.append(output)

        #outputs = torch.stack(outputs)
        #outputs = outputs.squeeze(1) 
        #out = F.conv2d(x1+x2,dk)
        return x6.mul(0.2) + x

class high_low_network(nn.Module):
    def __init__(self, in_nc, out_nc,nb=20, nf=32, kernel_size=3,norm_type='batch', act_type='leakyrelu'):
        super(high_low_network, self).__init__()

        self.conv0 = conv_block(in_nc, nf, kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
        rb_blocks = [base_block(nf, kernel_size=kernel_size, norm_type=norm_type,act_type=act_type) for _ in range(int(nb))]
        self.bb = sequential(*rb_blocks)
        
        self.conv1 = conv_block(nf,nf,kernel_size=kernel_size,norm_type=norm_type,act_type=act_type)
        self.up1 = pixelshuffle_block(nf, nf, upscale_factor=2)
        self.up2 = pixelshuffle_block(nf, nf, upscale_factor=2)
        self.conv2 = conv_block(nf,nf,kernel_size=kernel_size,norm_type=norm_type,act_type=act_type)
        self.conv3 = conv_block(nf,out_nc,kernel_size=kernel_size,norm_type=norm_type,act_type=act_type)
        #self.up3 = pixelshuffle_block(out_nc, out_nc, upscale_factor=4)
        #self.up3 = upconv_blcok(out_nc, out_nc, upscale_factor=2,mode='bicubic')


    def forward(self,x):
        #print(x.size())
        x0 = self.conv0(x)
        x1 = self.conv1(self.bb(x0))
        x2 = x1 + x0

        xu = self.up2(self.up1(x2))
        xr1 = self.conv3(self.conv2(xu))
        return xr1


class VGG_Block(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size=3,norm_type='batch', act_type='leakyrelu'):
        super(VGG_Block, self).__init__()

        self.conv0 = conv_block(in_nc, out_nc, kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
        self.conv1 = conv_block(out_nc, out_nc, kernel_size=kernel_size, stride=2, norm_type=None,act_type=act_type)

    def forward(self, x):
        x1 = self.conv0(x)
        out = self.conv1(x1)
        
        return out


class VGGGAPQualifier(nn.Module):
    def __init__(self, in_nc=3, base_nf=32, norm_type='batch', act_type='leakyrelu', mode='CNA'):
        super(VGGGAPQualifier, self).__init__()
        # 1024,768,3

        B11 = VGG_Block(in_nc,base_nf,norm_type=norm_type,act_type=act_type)
        # 512,384,32
        B12 = VGG_Block(base_nf,base_nf,norm_type=norm_type,act_type=act_type)
        # 256,192,32
        B13 = VGG_Block(base_nf,base_nf*2,norm_type=norm_type,act_type=act_type)
        # 128,96,64
        B14 = VGG_Block(base_nf*2,base_nf*2,norm_type=norm_type,act_type=act_type)
        # 64,48,64

        # 1024,768,3
        B21 = VGG_Block(in_nc,base_nf,norm_type=norm_type,act_type=act_type)
        # 512,384,32
        B22 = VGG_Block(base_nf,base_nf,norm_type=norm_type,act_type=act_type)
        # 256,192,32
        B23 = VGG_Block(base_nf,base_nf*2,norm_type=norm_type,act_type=act_type)
        # 128,96,64
        B24 = VGG_Block(base_nf*2,base_nf*2,norm_type=norm_type,act_type=act_type)
        # 64,48,64


        B3 = VGG_Block(base_nf*2,base_nf*4,norm_type=norm_type,act_type=act_type)
        # 32,24,128
        B4 = VGG_Block(base_nf*4,base_nf*8,norm_type=norm_type,act_type=act_type)
        # 16,12,256
        B5 = VGG_Block(base_nf*8,base_nf*16,norm_type=norm_type,act_type=act_type)
        
        self.feature1 = sequential(B11,B12,B13,B14)
        self.feature2 = sequential(B21,B22,B23,B24)

        self.combine = sequential(B3,B4,B5)
        self.gap = nn.AdaptiveAvgPool2d((1,1))

        # classifie
        self.classifier = nn.Sequential(
            nn.Linear(base_nf*16, 512), nn.LeakyReLU(0.2, True), nn.Dropout(0.25), nn.Linear(512,256),nn.LeakyReLU(0.2, True), nn.Dropout(0.5), nn.Linear(256, 1), nn.LeakyReLU(0.2, True))

    def forward(self, x):

        f1 = self.feature1(x)
        f2 = self.feature2(x)
        x = self.gap(self.combine(f1-f2))

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ResNetBlock(nn.Module):
    '''
    ResNet Block, 3-3 style
    with extra residual scaling used in EDSR
    (Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW 17)
    '''

    def __init__(self, in_nc, mid_nc, out_nc, kernel_size=3, stride=1, dilation=1, groups=1, \
            bias=True, pad_type='zero', norm_type=None, act_type='relu', mode='CNA', res_scale=1):
        super(ResNetBlock, self).__init__()
        conv0 = conv_block(in_nc, mid_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
            norm_type, act_type, mode)
        if mode == 'CNA':
            act_type = None
        if mode == 'CNAC':  # Residual path: |-CNAC-|
            act_type = None
            norm_type = None
        conv1 = conv_block(mid_nc, out_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
            norm_type, act_type, mode)
        # if in_nc != out_nc:
        #     self.project = conv_block(in_nc, out_nc, 1, stride, dilation, 1, bias, pad_type, \
        #         None, None)
        #     print('Need a projecter in ResNetBlock.')
        # else:
        #     self.project = lambda x:x
        self.res = sequential(conv0, conv1)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.res(x).mul(self.res_scale)
        return x + res

class ResidualDenseBlock_5C(nn.Module):
    '''
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = conv_block(nc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv2 = conv_block(nc+gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv3 = conv_block(nc+2*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv4 = conv_block(nc+3*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv5 = conv_block(nc+4*gc, nc, 3, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=last_act, mode=mode)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(0.2) + x


class RRDB(nn.Module):
    '''
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB2 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB3 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(0.2) + x




####################
# Upsampler
####################


def pixelshuffle_block(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True, \
                        pad_type='zero', norm_type=None, act_type='relu'):
    '''
    Pixel shuffle layer
    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
    Neural Network, CVPR17)
    '''
    conv = conv_block(in_nc, out_nc * (upscale_factor ** 2), kernel_size, stride, bias=bias, \
                        pad_type=pad_type, norm_type=None, act_type=None)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)

    n = norm(norm_type, out_nc) if norm_type else None
    a = act(act_type) if act_type else None
    return sequential(conv, pixel_shuffle, n, a)


def upconv_blcok(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True, \
                pad_type='zero', norm_type=None, act_type='relu', mode='nearest'):
    # Up conv
    # described in https://distill.pub/2016/deconv-checkerboard/
    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv = conv_block(in_nc, out_nc, kernel_size, stride, bias=bias, \
                        pad_type=pad_type, norm_type=norm_type, act_type=act_type)
    return sequential(upsample, conv)

def downconv_blcok(in_nc, out_nc, downscale_factor=2, kernel_size=3, stride=1, bias=True, \
                pad_type='zero', norm_type=None, act_type='relu', mode='nearest'):
    # Up conv
    # described in https://distill.pub/2016/deconv-checkerboard/
    f = 0.5
    upsample = nn.Upsample(scale_factor=f)
    conv = conv_block(in_nc, out_nc, kernel_size, stride, bias=bias, \
                        pad_type=pad_type, norm_type=norm_type, act_type=act_type)
    return sequential(upsample, conv)
