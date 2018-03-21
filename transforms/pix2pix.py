from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import types

class Compose(object):
  """Composes several transforms together.
  Args:
    transforms (List[Transform]): list of transforms to compose.
  Example:
    >>> transforms.Compose([
    >>>   transforms.CenterCrop(10),
    >>>   transforms.ToTensor(),
    >>> ])
  """
  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, imgA, imgB):
    for t in self.transforms:
      imgA, imgB = t(imgA, imgB)
    return imgA, imgB

class ToTensor(object):
  """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
  [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
  """
  def __call__(self, picA, picB):
    pics = [picA, picB]
    output = []
    for pic in pics: 
      if isinstance(pic, np.ndarray):
        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
      else:
        # handle PIL Image
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
          nchannel = 3
        else:
          nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float().div(255.)
      output.append(img)
    return output[0], output[1]

class ToPILImage(object):
  """Converts a torch.*Tensor of range [0, 1] and shape C x H x W
  or numpy ndarray of dtype=uint8, range[0, 255] and shape H x W x C
  to a PIL.Image of range [0, 255]
  """
  def __call__(self, picA, picB):
    pics = [picA, picB]
    output = []
    for pic in pics:
      npimg = pic
      mode = None
      if not isinstance(npimg, np.ndarray):
        npimg = pic.mul(255).byte().numpy()
        npimg = np.transpose(npimg, (1, 2, 0))

      if npimg.shape[2] == 1:
        npimg = npimg[:, :, 0]
        mode = "L"
      output.append(Image.fromarray(npimg, mode=mode))

    return output[0], output[1]

class Normalize(object):
  """Given mean: (R, G, B) and std: (R, G, B),
  will normalize each channel of the torch.*Tensor, i.e.
  channel = (channel - mean) / std
  """
  def __init__(self, mean, std):
    self.mean = mean
    self.std = std

  def __call__(self, tensorA, tensorB):
    tensors = [tensorA, tensorB]
    output = []
    for tensor in tensors:
      # TODO: make efficient
      for t, m, s in zip(tensor, self.mean, self.std):
        t.sub_(m).div_(s)
      output.append(tensor)
    return output[0], output[1]

class Scale(object):
  """Rescales the input PIL.Image to the given 'size'.
  'size' will be the size of the smaller edge.
  For example, if height > width, then image will be
  rescaled to (size * height / width, size)
  size: size of the smaller edge
  interpolation: Default: PIL.Image.BILINEAR
  """
  def __init__(self, size, interpolation=Image.BILINEAR):
    self.size = size
    self.interpolation = interpolation

  def __call__(self, imgA, imgB):
    imgs = [imgA, imgB]
    output = []
    for img in imgs:
      w, h = img.size
      if (w <= h and w == self.size) or (h <= w and h == self.size):
        output.append(img)
        continue
      if w < h:
        ow = self.size
        oh = int(self.size * h / w)
        output.append(img.resize((ow, oh), self.interpolation))
        continue
      else:
        oh = self.size
        ow = int(self.size * w / h)
      output.append(img.resize((ow, oh), self.interpolation))
    return output[0], output[1]

class CenterCrop(object):
  """Crops the given PIL.Image at the center to have a region of
  the given size. size can be a tuple (target_height, target_width)
  or an integer, in which case the target will be of a square shape (size, size)
  """
  def __init__(self, size):
    if isinstance(size, numbers.Number):
      self.size = (int(size), int(size))
    else:
      self.size = size

  def __call__(self, imgA, imgB):
    imgs = [imgA, imgB]
    output = []
    for img in imgs:
      w, h = img.size
      th, tw = self.size
      x1 = int(round((w - tw) / 2.))
      y1 = int(round((h - th) / 2.))
      output.append(img.crop((x1, y1, x1 + tw, y1 + th)))
    return output[0], output[1]

class Pad(object):
  """Pads the given PIL.Image on all sides with the given "pad" value"""
  def __init__(self, padding, fill=0):
    assert isinstance(padding, numbers.Number)
    assert isinstance(fill, numbers.Number) or isinstance(fill, str) or isinstance(fill, tuple)
    self.padding = padding
    self.fill = fill

  def __call__(self, imgA, imgB):
    imgs = [imgA, imgB]
    output = []
    for img in imgs:
      output.append(ImageOps.expand(img, border=self.padding, fill=self.fill))
    return output[0], output[1]

class Lambda(object):
  """Applies a lambda as a transform."""
  def __init__(self, lambd):
    assert isinstance(lambd, types.LambdaType)
    self.lambd = lambd

  def __call__(self, imgA, imgB):
    imgs = [imgA, imgB]
    output = []
    for img in imgs:
      output.append(self.lambd(img))
    return output[0], output[1]

class RandomCrop(object):
  """Crops the given PIL.Image at a random location to have a region of
  the given size. size can be a tuple (target_height, target_width)
  or an integer, in which case the target will be of a square shape (size, size)
  """
  def __init__(self, size, padding=0):
    if isinstance(size, numbers.Number):
      self.size = (int(size), int(size))
    else:
      self.size = size
    self.padding = padding

  def __call__(self, imgA, imgB):
    imgs = [imgA, imgB]
    output = []
    x1 = -1
    y1 = -1
    for img in imgs:
      if self.padding > 0:
        img = ImageOps.expand(img, border=self.padding, fill=0)

      w, h = img.size
      th, tw = self.size
      if w == tw and h == th:
        output.append(img)
        continue

      if x1 == -1 and y1 == -1:
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
      output.append(img.crop((x1, y1, x1 + tw, y1 + th)))
    return output[0], output[1]

class RandomHorizontalFlip(object):
  """Randomly horizontally flips the given PIL.Image with a probability of 0.5
  """
  def __call__(self, imgA, imgB):
    imgs = [imgA, imgB]
    output = []
    flag = random.random() < 0.5
    for img in imgs:
      if flag:
        output.append(img.transpose(Image.FLIP_LEFT_RIGHT))
      else:
        output.append(img)
    return output[0], output[1]
