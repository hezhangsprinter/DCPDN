import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torchvision.models as models
from torch.autograd import Variable



def conv_block(in_dim,out_dim):
  return nn.Sequential(nn.Conv2d(in_dim,in_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.Conv2d(in_dim,in_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.Conv2d(in_dim,out_dim,kernel_size=1,stride=1,padding=0),
                       nn.AvgPool2d(kernel_size=2,stride=2))
def deconv_block(in_dim,out_dim):
  return nn.Sequential(nn.Conv2d(in_dim,out_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.Conv2d(out_dim,out_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.UpsamplingNearest2d(scale_factor=2))


def blockUNet1(in_c, out_c, name, transposed=False, bn=False, relu=True, dropout=False):
  block = nn.Sequential()
  if relu:
    block.add_module('%s.relu' % name, nn.ReLU(inplace=True))
  else:
    block.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
  if not transposed:
    block.add_module('%s.conv' % name, nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False))
  else:
    block.add_module('%s.tconv' % name, nn.ConvTranspose2d(in_c, out_c, 3, 1, 1, bias=False))
  if bn:
    block.add_module('%s.bn' % name, nn.BatchNorm2d(out_c))
  if dropout:
    block.add_module('%s.dropout' % name, nn.Dropout2d(0.5, inplace=True))
  return block

def blockUNet(in_c, out_c, name, transposed=False, bn=False, relu=True, dropout=False):
  block = nn.Sequential()
  if relu:
    block.add_module('%s.relu' % name, nn.ReLU(inplace=True))
  else:
    block.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
  if not transposed:
    block.add_module('%s.conv' % name, nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False))
  else:
    block.add_module('%s.tconv' % name, nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False))
  if bn:
    block.add_module('%s.bn' % name, nn.BatchNorm2d(out_c))
  if dropout:
    block.add_module('%s.dropout' % name, nn.Dropout2d(0.5, inplace=True))
  return block


class D1(nn.Module):
  def __init__(self, nc, ndf, hidden_size):
    super(D1, self).__init__()

    # 256
    self.conv1 = nn.Sequential(nn.Conv2d(nc,ndf,kernel_size=3,stride=1,padding=1),
                               nn.ELU(True))
    # 256
    self.conv2 = conv_block(ndf,ndf)
    # 128
    self.conv3 = conv_block(ndf, ndf*2)
    # 64
    self.conv4 = conv_block(ndf*2, ndf*3)
    # 32
    self.encode = nn.Conv2d(ndf*3, hidden_size, kernel_size=1,stride=1,padding=0)
    self.decode = nn.Conv2d(hidden_size, ndf, kernel_size=1,stride=1,padding=0)
    # 32
    self.deconv4 = deconv_block(ndf, ndf)
    # 64
    self.deconv3 = deconv_block(ndf, ndf)
    # 128
    self.deconv2 = deconv_block(ndf, ndf)
    # 256
    self.deconv1 = nn.Sequential(nn.Conv2d(ndf,ndf,kernel_size=3,stride=1,padding=1),
                                 nn.ELU(True),
                                 nn.Conv2d(ndf,ndf,kernel_size=3,stride=1,padding=1),
                                 nn.ELU(True),
                                 nn.Conv2d(ndf, nc, kernel_size=3, stride=1, padding=1),
                                 nn.Tanh())
    """
    self.deconv1 = nn.Sequential(nn.Conv2d(ndf,nc,kernel_size=3,stride=1,padding=1),
                                 nn.Tanh())
    """
  def forward(self,x):
    out1 = self.conv1(x)
    out2 = self.conv2(out1)
    out3 = self.conv3(out2)
    out4 = self.conv4(out3)
    out5 = self.encode(out4)
    dout5= self.decode(out5)
    dout4= self.deconv4(dout5)
    dout3= self.deconv3(dout4)
    dout2= self.deconv2(dout3)
    dout1= self.deconv1(dout2)
    return dout1

class D(nn.Module):
  def __init__(self, nc, nf):
    super(D, self).__init__()

    main = nn.Sequential()
    # 256
    layer_idx = 1
    name = 'layer%d' % layer_idx
    main.add_module('%s.conv' % name, nn.Conv2d(nc, nf, 4, 2, 1, bias=False))

    # 128
    layer_idx += 1
    name = 'layer%d' % layer_idx
    main.add_module(name, blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False))

    # 64
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module(name, blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False))

    # 32
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    main.add_module('%s.conv' % name, nn.Conv2d(nf, nf*2, 4, 1, 1, bias=False))
    main.add_module('%s.bn' % name, nn.BatchNorm2d(nf*2))

    # 31
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    main.add_module('%s.conv' % name, nn.Conv2d(nf, 1, 4, 1, 1, bias=False))
    main.add_module('%s.sigmoid' % name , nn.Sigmoid())
    # 30 (sizePatchGAN=30)

    self.main = main

  def forward(self, x):
    output = self.main(x)
    return output


class D_tran(nn.Module):
  def __init__(self, nc, nf):
    super(D_tran, self).__init__()

    main = nn.Sequential()
    # 256
    layer_idx = 1
    name = 'layer%d' % layer_idx
    main.add_module('%s.conv' % name, nn.Conv2d(nc, nf, 4, 2, 1, bias=False))

    # 128
    layer_idx += 1
    name = 'layer%d' % layer_idx
    main.add_module(name, blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False))

    # 64
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module(name, blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False))

    # 32
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    main.add_module('%s.conv' % name, nn.Conv2d(nf, nf*2, 4, 1, 1, bias=False))
    main.add_module('%s.bn' % name, nn.BatchNorm2d(nf*2))

    # 31
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    main.add_module('%s.conv' % name, nn.Conv2d(nf, 1, 4, 1, 1, bias=False))
    main.add_module('%s.sigmoid' % name , nn.Sigmoid())
    # 30 (sizePatchGAN=30)

    self.main = main

  def forward(self, x):
    output = self.main(x)
    return output



class G(nn.Module):
  def __init__(self, input_nc, output_nc, nf):
    super(G, self).__init__()
    # input is 256 x 256
    layer_idx = 1
    name = 'layer%d' % layer_idx
    layer1 = nn.Sequential()
    layer1.add_module(name, nn.Conv2d(input_nc, nf, 4, 2, 1, bias=False))
    # input is 128 x 128
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer2 = blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 64 x 64
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer3 = blockUNet(nf*2, nf*4, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 32
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer4 = blockUNet(nf*4, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 16
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer5 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 8
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer6 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 4
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer7 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 2 x  2
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer8 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)

    ## NOTE: decoder
    # input is 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8
    dlayer8 = blockUNet(d_inc, nf*8, name, transposed=True, bn=False, relu=True, dropout=True)

    # input is 2
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8*2
    dlayer7 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=True)
    # input is 4
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8*2
    dlayer6 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=True)
    # input is 8
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8*2
    dlayer5 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=False)
    # input is 16
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8*2
    dlayer4 = blockUNet(d_inc, nf*4, name, transposed=True, bn=True, relu=True, dropout=False)
    # input is 32
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*4*2
    dlayer3 = blockUNet(d_inc, nf*2, name, transposed=True, bn=True, relu=True, dropout=False)
    # input is 64
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*2*2
    dlayer2 = blockUNet(d_inc, nf, name, transposed=True, bn=True, relu=True, dropout=False)
    # input is 128
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    dlayer1 = nn.Sequential()
    d_inc = nf*2
    dlayer1.add_module('%s.relu' % name, nn.ReLU(inplace=True))
    dlayer1.add_module('%s.tconv' % name, nn.ConvTranspose2d(d_inc, 20, 4, 2, 1, bias=False))

    dlayerfinal = nn.Sequential()

    dlayerfinal.add_module('%s.conv' % name, nn.Conv2d(24, output_nc, 3, 1, 1, bias=False))
    dlayerfinal.add_module('%s.tanh' % name, nn.Tanh())

    self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
    self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
    self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
    self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

    self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)

    self.upsample = F.upsample_nearest

    self.layer1 = layer1
    self.layer2 = layer2
    self.layer3 = layer3
    self.layer4 = layer4
    self.layer5 = layer5
    self.layer6 = layer6
    self.layer7 = layer7
    self.layer8 = layer8
    self.dlayer8 = dlayer8
    self.dlayer7 = dlayer7
    self.dlayer6 = dlayer6
    self.dlayer5 = dlayer5
    self.dlayer4 = dlayer4
    self.dlayer3 = dlayer3
    self.dlayer2 = dlayer2
    self.dlayer1 = dlayer1
    self.dlayerfinal = dlayerfinal
    self.relu=nn.LeakyReLU(0.2, inplace=True)

  def forward(self, x):
    out1 = self.layer1(x)
    out2 = self.layer2(out1)
    out3 = self.layer3(out2)
    out4 = self.layer4(out3)
    out5 = self.layer5(out4)
    out6 = self.layer6(out5)
    out7 = self.layer7(out6)
    out8 = self.layer8(out7)
    dout8 = self.dlayer8(out8)
    dout8_out7 = torch.cat([dout8, out7], 1)
    dout7 = self.dlayer7(dout8_out7)
    dout7_out6 = torch.cat([dout7, out6], 1)
    dout6 = self.dlayer6(dout7_out6)
    dout6_out5 = torch.cat([dout6, out5], 1)
    dout5 = self.dlayer5(dout6_out5)
    dout5_out4 = torch.cat([dout5, out4], 1)
    dout4 = self.dlayer4(dout5_out4)
    dout4_out3 = torch.cat([dout4, out3], 1)
    dout3 = self.dlayer3(dout4_out3)
    dout3_out2 = torch.cat([dout3, out2], 1)
    dout2 = self.dlayer2(dout3_out2)
    dout2_out1 = torch.cat([dout2, out1], 1)
    dout1 = self.dlayer1(dout2_out1)

    shape_out = dout1.data.size()
    # print(shape_out)
    shape_out = shape_out[2:4]

    x101 = F.avg_pool2d(dout1, 16)
    x102 = F.avg_pool2d(dout1, 8)
    x103 = F.avg_pool2d(dout1, 4)
    x104 = F.avg_pool2d(dout1, 2)

    x1010 = self.upsample(self.relu(self.conv1010(x101)),size=shape_out)
    x1020 = self.upsample(self.relu(self.conv1020(x102)),size=shape_out)
    x1030 = self.upsample(self.relu(self.conv1030(x103)),size=shape_out)
    x1040 = self.upsample(self.relu(self.conv1040(x104)),size=shape_out)

    dehaze = torch.cat((x1010, x1020, x1030, x1040, dout1), 1)

    dout1 = self.dlayerfinal(dehaze)

    return dout1

class G2(nn.Module):
  def __init__(self, input_nc, output_nc, nf):
    super(G2, self).__init__()
    # input is 256 x 256
    layer_idx = 1
    name = 'layer%d' % layer_idx
    layer1 = nn.Sequential()
    layer1.add_module(name, nn.Conv2d(input_nc, nf, 4, 2, 1, bias=False))
    # input is 128 x 128
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer2 = blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 64 x 64
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer3 = blockUNet(nf*2, nf*4, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 32
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer4 = blockUNet(nf*4, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 16
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer5 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 8
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer6 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 4
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer7 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 2 x  2
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer8 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)

    ## NOTE: decoder
    # input is 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8
    dlayer8 = blockUNet(d_inc, nf*8, name, transposed=True, bn=False, relu=True, dropout=True)

    #import pdb; pdb.set_trace()
    # input is 2
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8*2
    dlayer7 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=True)
    # input is 4
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8*2
    dlayer6 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=True)
    # input is 8
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8*2
    dlayer5 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=False)
    # input is 16
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8*2
    dlayer4 = blockUNet(d_inc, nf*4, name, transposed=True, bn=True, relu=True, dropout=False)
    # input is 32
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*4*2
    dlayer3 = blockUNet(d_inc, nf*2, name, transposed=True, bn=True, relu=True, dropout=False)
    # input is 64
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*2*2
    dlayer2 = blockUNet(d_inc, nf, name, transposed=True, bn=True, relu=True, dropout=False)
    # input is 128
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    dlayer1 = nn.Sequential()
    d_inc = nf*2
    dlayer1.add_module('%s.relu' % name, nn.ReLU(inplace=True))
    dlayer1.add_module('%s.tconv' % name, nn.ConvTranspose2d(d_inc, output_nc, 4, 2, 1, bias=False))
    dlayer1.add_module('%s.tanh' % name, nn.LeakyReLU(0.2, inplace=True))

    self.layer1 = layer1
    self.layer2 = layer2
    self.layer3 = layer3
    self.layer4 = layer4
    self.layer5 = layer5
    self.layer6 = layer6
    self.layer7 = layer7
    self.layer8 = layer8
    self.dlayer8 = dlayer8
    self.dlayer7 = dlayer7
    self.dlayer6 = dlayer6
    self.dlayer5 = dlayer5
    self.dlayer4 = dlayer4
    self.dlayer3 = dlayer3
    self.dlayer2 = dlayer2
    self.dlayer1 = dlayer1

  def forward(self, x):
    out1 = self.layer1(x)
    out2 = self.layer2(out1)
    out3 = self.layer3(out2)
    out4 = self.layer4(out3)
    out5 = self.layer5(out4)
    out6 = self.layer6(out5)
    out7 = self.layer7(out6)
    out8 = self.layer8(out7)
    dout8 = self.dlayer8(out8)
    dout8_out7 = torch.cat([dout8, out7], 1)
    dout7 = self.dlayer7(dout8_out7)
    dout7_out6 = torch.cat([dout7, out6], 1)
    dout6 = self.dlayer6(dout7_out6)
    dout6_out5 = torch.cat([dout6, out5], 1)
    dout5 = self.dlayer5(dout6_out5)
    dout5_out4 = torch.cat([dout5, out4], 1)
    dout4 = self.dlayer4(dout5_out4)
    dout4_out3 = torch.cat([dout4, out3], 1)
    dout3 = self.dlayer3(dout4_out3)
    dout3_out2 = torch.cat([dout3, out2], 1)
    dout2 = self.dlayer2(dout3_out2)
    dout2_out1 = torch.cat([dout2, out1], 1)
    dout1 = self.dlayer1(dout2_out1)
    return dout1


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.upsample_nearest(out, scale_factor=2)






class Dense(nn.Module):
    def __init__(self):
        super(Dense, self).__init__()




        ############# 256-256  ##############
        haze_class = models.densenet121(pretrained=True)

        self.conv0=haze_class.features.conv0
        self.norm0=haze_class.features.norm0
        self.relu0=haze_class.features.relu0
        self.pool0=haze_class.features.pool0

        ############# Block1-down 64-64  ##############
        self.dense_block1=haze_class.features.denseblock1
        self.trans_block1=haze_class.features.transition1

        ############# Block2-down 32-32  ##############
        self.dense_block2=haze_class.features.denseblock2
        self.trans_block2=haze_class.features.transition2

        ############# Block3-down  16-16 ##############
        self.dense_block3=haze_class.features.denseblock3
        self.trans_block3=haze_class.features.transition3

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock(512,256)
        self.trans_block4=TransitionBlock(768,128)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock(384,256)
        self.trans_block5=TransitionBlock(640,128)

        ############# Block6-up 32-32   ##############
        self.dense_block6=BottleneckBlock(256,128)
        self.trans_block6=TransitionBlock(384,64)


        ############# Block7-up 64-64   ##############
        self.dense_block7=BottleneckBlock(64,64)
        self.trans_block7=TransitionBlock(128,32)

        ## 128 X  128
        ############# Block8-up c  ##############
        self.dense_block8=BottleneckBlock(32,32)
        self.trans_block8=TransitionBlock(64,16)

        self.conv_refin=nn.Conv2d(19,20,3,1,1)
        self.tanh=nn.Tanh()


        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)
        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        ## 256x256
        x0=self.pool0(self.relu0(self.norm0(self.conv0(x))))

        ## 64 X 64
        x1=self.dense_block1(x0)
        # print x1.size()
        x1=self.trans_block1(x1)

        ###  32x32
        x2=self.trans_block2(self.dense_block2(x1))
        # print  x2.size()


        ### 16 X 16
        x3=self.trans_block3(self.dense_block3(x2))

        # x3=Variable(x3.data,requires_grad=True)

        ## 8 X 8
        x4=self.trans_block4(self.dense_block4(x3))

        x42=torch.cat([x4,x2],1)
        ## 16 X 16
        x5=self.trans_block5(self.dense_block5(x42))

        x52=torch.cat([x5,x1],1)
        ##  32 X 32
        x6=self.trans_block6(self.dense_block6(x52))

        ##  64 X 64
        x7=self.trans_block7(self.dense_block7(x6))

        ##  128 X 128
        x8=self.trans_block8(self.dense_block8(x7))

        # print x8.size()
        # print x.size()

        x8=torch.cat([x8,x],1)

        # print x8.size()

        x9=self.relu(self.conv_refin(x8))

        shape_out = x9.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(x9, 32)
        x102 = F.avg_pool2d(x9, 16)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 4)

        x1010 = self.upsample(self.relu(self.conv1010(x101)), size=shape_out)
        x1020 = self.upsample(self.relu(self.conv1020(x102)), size=shape_out)
        x1030 = self.upsample(self.relu(self.conv1030(x103)), size=shape_out)
        x1040 = self.upsample(self.relu(self.conv1040(x104)), size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        dehaze = self.tanh(self.refine3(dehaze))

        return dehaze



class dehaze(nn.Module):
  def __init__(self, input_nc, output_nc, nf):
    super(dehaze, self).__init__()
    self.tran_est=G(input_nc=3,output_nc=3, nf=64)
    self.atp_est=G2(input_nc=3,output_nc=3, nf=8)

    self.tran_dense=Dense()
    self.relu=nn.LeakyReLU(0.2, inplace=True)
    # self.relu5=nn.ReLU6()

    self.tanh=nn.Tanh()

    self.refine1= nn.Conv2d(6, 20, kernel_size=3,stride=1,padding=1)
    self.refine2= nn.Conv2d(20, 20, kernel_size=3,stride=1,padding=1)
    self.threshold=nn.Threshold(0.1, 0.1)

    self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
    self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
    self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
    self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

    self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)

    self.upsample = F.upsample_nearest

    self.batch1= nn.BatchNorm2d(20)

    # self.batch2 = nn.InstanceNorm2d(100, affine=True)
  def forward(self, x):


    tran=self.tran_dense(x)
    atp= self.atp_est(x)


    # x = Variable(x.data, requires_grad=True)

    # zz= torch.abs(self.threshold(tran))
    zz= torch.abs((tran))+(10**-10)
    shape_out1 = atp.data.size()
    # print(shape_out)
    # shape_out = shape_out[0:5]

    # atp_mean=torch.mean(atp)

    # threshold = nn.Threshold(10, 0.95)

    shape_out = shape_out1[2:4]
    atp = F.avg_pool2d(atp, shape_out1[2])
    atp = self.upsample(self.relu(atp),size=shape_out)


    # print atp.data
    # atp = threshold(atp)



    dehaze= (x-atp)/zz+ atp
    dehaze2=dehaze


    dehaze=torch.cat([dehaze,x],1)
    # dehaze=dehaze/(tran+(10**-10))
    # dehaze=self.relu(self.batch1(self.refine1(dehaze)))
    # dehaze=self.relu(self.batch1(self.refine2(dehaze)))

    dehaze=self.relu((self.refine1(dehaze)))
    dehaze=self.relu((self.refine2(dehaze)))
    shape_out = dehaze.data.size()
    # print(shape_out)
    shape_out = shape_out[2:4]

    x101 = F.avg_pool2d(dehaze, 32)
    x1010 = F.avg_pool2d(dehaze, 32)

    x102 = F.avg_pool2d(dehaze, 16)
    x1020 = F.avg_pool2d(dehaze, 16)

    x103 = F.avg_pool2d(dehaze, 8)

    x104 = F.avg_pool2d(dehaze, 4)


    x1010 = self.upsample(self.relu(self.conv1010(x101)),size=shape_out)
    x1020 = self.upsample(self.relu(self.conv1020(x102)),size=shape_out)
    x1030 = self.upsample(self.relu(self.conv1030(x103)),size=shape_out)
    x1040 = self.upsample(self.relu(self.conv1040(x104)),size=shape_out)

    dehaze = torch.cat((x1010, x1020, x1030, x1040, dehaze), 1)
    dehaze= self.tanh(self.refine3(dehaze))

    return dehaze, tran, atp, dehaze2