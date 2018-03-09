# DCPDN

## Densely Connected Pyramid Dehazing Network (CVPR'2018)
[He Zhang](https://sites.google.com/site/hezhangsprinter), [Vishal M. Patel](http://www.rci.rutgers.edu/~vmp93/)

[[Paper Link](https://arxiv.org/abs/1802.07412)] (CVPR'18)

We present a novel density-aware multi-stream densely connected convolutional neural
network-based algorithm, called DID-MDN, for joint rain density estimation and de-raining. The proposed method
enables the network itself to automatically determine the rain-density information and then efficiently remove the
corresponding rain-streaks guided by the estimated rain-density label. To better characterize rain-streaks with dif-
ferent scales and shapes, a multi-stream densely connected de-raining network is proposed which efficiently leverages
features from different scales. Furthermore, a new dataset containing images with rain-density labels is created and
used to train the proposed density-aware network. 

	@inproceedings{dehaze_zhang_2018,		
	  title={Densely Connected Pyramid Dehazing Network},
	  author={Zhang, He and Patel, Vishal M},
	  booktitle={CVPR},
	  year={2018}
	} 

<p align="center">
<img src="demo_image/over_input1.png" width="250px" height="200px"/>         <img src="demo_image/over_our.png" width="250px" height="200px"/>



## Prerequisites:
1. Linux
2. Python 2 or 3
3. CPU or NVIDIA GPU + CUDA CuDNN (CUDA 8.0)
 
## Installation:
1. Install PyTorch and dependencies from http://pytorch.org (Ubuntu+Python2.7)
   (conda install pytorch torchvision -c pytorch)

2. Install Torch vision from the source.
   
   	git clone https://github.com/pytorch/vision
	
   	cd vision
	
	python setup.py install

3. Install python package: 
   numpy, scipy, PIL, pdb
   
## Demo using pre-trained model
	python demo.py --dataroot ./facades/nat_new4 --valDataroot ./facades/nat_new4 --netG ./demo_model/netG_epoch_8.pth   
Pre-trained dehazing model can be downloaded at (put it in the folder 'demo_model'): https://drive.google.com/drive/folders/1BmNP5ZUWEFeGGEL1NsZSRbYPyjBQ7-nn?usp=sharing

Testing images (nature)  can be downloaded at (put it in the folder 'facades'):
https://drive.google.com/drive/folders/1q5bRQGgS8SFEGqMwrLlku4Ad-0Tn3va7?usp=sharing

Testing images (syn (Test A in the paper))  can be downloaded at (put it in the folder 'facades'):
https://drive.google.com/drive/folders/1hbwYCzoI3R3o2Gj_kfT6GHG7RmYEOA-P?usp=sharing


## Training
	python train.py --dataroot ./train512 --valDataroot ./val512 --exp ./checkpoints/
## Testing
	python demo.py --dataroot ./your_dataroot --valDataroot ./your_dataroot --netG ./pre_trained/netG_epoch_9.pth   
## Dataset


Training images (syn)  can be downloaded at (put it in the folder 'facades'):
https://drive.google.com/drive/folders/1hbwYCzoI3R3o2Gj_kfT6GHG7RmYEOA-P?usp=sharing

All the syn samples (both training and testing) are strored in Hdf5 file.
Following are the sample python codes how to read the Hdf5 file:
    
    file_name=self.root+'/'+str(index)+'.h5'
    f=h5py.File(file_name,'r')

    haze_image=f['haze'][:]
    gt_trans_map=f['trans'][:]
    gt_ato_map=f['ato'][:]
    GT=f['gt'][:]

Testing images (nature)  can be downloaded at (put it in the folder 'facades'):
https://drive.google.com/drive/folders/1q5bRQGgS8SFEGqMwrLlku4Ad-0Tn3va7?usp=sharing

Testing images (syn (Test A in the paper))  can be downloaded at (put it in the folder 'facades'):
https://drive.google.com/drive/folders/1hbwYCzoI3R3o2Gj_kfT6GHG7RmYEOA-P?usp=sharing

## Acknowledgments

Great thanks for the insight discussion with [Vishwanath Sindagi](http://www.vishwanathsindagi.com/) and initial discussion with [Dr. Kevin S. Zhou](https://sites.google.com/site/skevinzhou/home)
