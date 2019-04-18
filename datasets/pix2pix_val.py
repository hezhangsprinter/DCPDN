import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
import h5py
import glob
import pdb

IMG_EXTENSIONS = [
  '.jpg', '.JPG', '.jpeg', '.JPEG',
  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
  return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
  images = []
  if not os.path.isdir(dir):
    raise Exception('Check dataroot')
  for root, _, fnames in sorted(os.walk(dir)):
    for fname in fnames:
      if is_image_file(fname):
        path = os.path.join(dir, fname)
        item = path
        images.append(item)
  return images

def default_loader(path):
  return Image.open(path).convert('RGB')

class pix2pix_val(data.Dataset):
  def __init__(self, root, transform=None, loader=default_loader, seed=None):
    # imgs = make_dataset(root)
    # if len(imgs) == 0:
    #   raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
    #              "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
    self.root = root
    # self.imgs = imgs
    self.transform = transform
    self.loader = loader
    self.paths = glob.glob(self.root + '/*h5')
    
    # self.sampler = SequentialSampler(dataset)

    if seed is not None:
      np.random.seed(seed)
  # def __getitem__(self, _):
  def __getitem__(self, index):

    # index = np.random.randint(1,self.__len__())
    # index = np.random.randint(self.__len__(), size=1)[0]

    # path = self.imgs[index]
    # img = self.loader(path)
    #img = img.resize((w, h), Image.BILINEAR)



    file_name=self.root+'/'+str(index)+'.h5'
    f=h5py.File(file_name,'r')
    haze_image=f['haze'][:]
    trans_map=f['trans'][:]
    ato_map=f['ato'][:]
    GT=f['gt'][:]



    haze_image=np.swapaxes(haze_image,0,2)
    trans_map=np.swapaxes(trans_map,0,2)
    ato_map=np.swapaxes(ato_map,0,2)
    GT=np.swapaxes(GT,0,2)



    haze_image=np.swapaxes(haze_image,1,2)
    trans_map=np.swapaxes(trans_map,1,2)
    ato_map=np.swapaxes(ato_map,1,2)
    GT=np.swapaxes(GT,1,2)

    # if np.random.uniform()>0.5:
    #   haze_image=np.flip(haze_image,2).copy()

    return haze_image, GT,  trans_map, ato_map, file_name[len(self.root)+1:-3]

  def __len__(self):
    train_list=glob.glob(self.root+'/*h5')
    # print len(train_list)
    return len(train_list)
