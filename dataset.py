# -*- coding: utf-8 -*-

import os
import numpy as np
import torch as tc
from torch.utils.data import Dataset

import cv2
import imgaug.augmenters as iaa


def get_aug_seq(index=1):
    if index == 1:
        return iaa.Sequential( [iaa.Fliplr(0.5),
                                iaa.Flipud(0.5),
                                ], random_order=True)
    elif index == 2:
        return iaa.Sequential([iaa.Fliplr(0.5),
                                iaa.Flipud(0.5),
                                iaa.Sometimes(0.5, iaa.Crop(percent=(0, 0.05))),
                                iaa.Affine(
                                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                                    rotate=(-25, 25),
                                    shear=(-8, 8))
                                ], random_order=True)
    elif index == 3:
        return iaa.Sequential([iaa.MultiplyBrightness((0.8, 1.2)),
                                iaa.MultiplyHue((0.5, 1.5)),
                                iaa.MultiplySaturation((0.5, 1.5)),
                                iaa.Fliplr(0.5),
                                iaa.Flipud(0.5),
                                iaa.Sometimes(
                                    0.5,
                                    iaa.GaussianBlur(sigma=(0, 0.5))
                                ),
                                iaa.Sometimes(0.5, iaa.Crop(percent=(0, 0.1))),
                                iaa.LinearContrast((0.75, 1.5)),
                                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                                iaa.Affine(
                                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                                    rotate=(-25, 25),
                                    shear=(-8, 8))
                                ], random_order=True)
    elif index == 4:
        return iaa.Sequential([iaa.Fliplr(0.5),
                               iaa.Flipud(0.5),
                               iaa.MultiplyBrightness((0.8, 1.2)),
                                iaa.MultiplyHue((0.5, 1.5)),
                                iaa.MultiplySaturation((0.5, 1.5))
                                ], random_order=True)
    elif index == 5:
        return iaa.Sequential([])

    elif index == 6:
        return iaa.Sequential([iaa.Fliplr(0.5),
                                iaa.Sometimes(
                                0.5,
                                iaa.GaussianBlur(sigma=(0, 0.5))
                                ),
                                iaa.LinearContrast((0.75, 1.5)),
                                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                                iaa.Multiply((0.8, 1.2), per_channel=0.5),
                                iaa.Affine(
                                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                                    rotate=(-5, 5),
                                    shear=(-10, 10))
                                ], random_order=True)
    else:
        raise ValueError('Augmenter not included')


#预处理与数据增强
class AugDataset(Dataset):
    def __init__(self, data_dir, val=False, size=224, aug=1):
        self.dir = data_dir
        self.val = val
        self.size = size
        assert os.path.exists(self.dir), f'Directory {self.dir} not exist'
        #截断指令
        self.ids = []
        self.targets = []
        dirlist = self.getsubdir(self.dir)
        print("dirlist:",dirlist)
        for d in dirlist:
            print('d:',d)
            subdirlist = os.listdir(d)
            print('subdirlist:',subdirlist)
            #subdirlist是1类下 所有图片组成的列表
            self.ids += [os.path.join(d, f) for f in subdirlist
                         if os.path.isfile(os.path.join(d, f))]
            print('self.ids:',self.ids)
            try:
                target_name = int(d[-1]) - 1
            except:
              raise TypeError(f'directory name should not be {d}')
            self.targets += [target_name]*len(subdirlist)
        self.seq = get_aug_seq(aug)
        self.means = [0.485, 0.456, 0.406]
        self.stds = [0.229, 0.224, 0.225]
        if val:
            print(f'Creating validation dataset with {len(self.ids)} examples')
        else:
            print(f'Creating training dataset with {len(self.ids)} examples & augmentation {aug}')

    '''
    重载len
    '''
    def __len__(self):
        return len(self.ids)
    
    @classmethod
    def augmentation(cls, ori_img, seq, size):
        image = cv2.resize(ori_img, (size, size), interpolation=cv2.INTER_NEAREST)
        img_aug = seq(image=image)
        img_aug = img_aug / 255
        img_aug = img_aug.transpose((2, 0, 1))#图像反转
        return img_aug

    '''
    图片尺寸归一化、像素归一化；
    '''
    @classmethod
    def resize(cls, ori_img, size):
        image = cv2.resize(ori_img, (size, size), interpolation=cv2.INTER_NEAREST)
        image = image / 255
        image = image.transpose((2, 0, 1))
        return image
    
    @classmethod
    def getsubdir(cls, root_dir):
        return [os.path.join(root_dir, dI) for dI in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, dI))]


    '''
    openCV读取数据
    cv2.imdecode()函数从指定的内存缓存中读取数据，并把数据转换(解码)成图像格式;主要用于从网络传输数据中恢复出图像。
    '''
    @classmethod
    def cv_imread(cls, filepath):
        cv_img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), -1)
        return cv_img
    
    '''
    将图像变成灰度图，用掩膜处理
    '''
    @classmethod
    def crop_image_from_gray(cls, img,tol=7):
        if img.ndim ==2:#判断是否为2维数组(即为灰度图像)
            mask = img>tol
            return img[np.ix_(mask.any(1),mask.any(0))]
        elif img.ndim==3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            mask = gray_img>tol
            
            check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
            if (check_shape == 0): # image is too dark so that we crop out everything,
                return img # return original image
            else:
                img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
                img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
                img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
                img = np.stack([img1,img2,img3],axis=-1)
            return img
    
    @classmethod
    def load_ben_color(cls, image, sigmaX=30):
        image = cls.crop_image_from_gray(image)
        image = cv2.resize(image, (512, 512))
        image = cv2.addWeighted (image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
        #cv2.addWeighted() 图像融合。这里是把图像与高斯模糊后的图像做融合
        return image

    '''
    重载getitem
    x[i]可是为x.__getitem__(x,i)
    '''
    def __getitem__(self, i):
        file = self.ids[i]
        target = self.targets[i]
        try:
            data = self.cv_imread(file)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        except:
            raise RuntimeError(f'{file} not able to be loaded')
        
        data = self.load_ben_color(data)

        if self.val:
            data = self.resize(data, self.size)
        else:
            data = self.augmentation(data, self.seq, self.size)
        
        data = tc.from_numpy(data)
        
        for t, m, s in zip(data, self.means, self.stds):
            t.sub_(m).div_(s)
        return data, target

#简易预处理，经过处理输出在0-1之间
class OriginDataset(Dataset):
    def __init__(self, data_dir, val=False, size=224):
        self.dir = data_dir
        self.val = val
        self.size = size
        assert os.path.exists(self.dir), f'Directory {self.dir} not exist'
        # 截断指令
        self.ids = []
        self.targets = []
        dirlist = self.getsubdir(self.dir)
        print("dirlist:", dirlist)
        for d in dirlist:
            print('d:', d)
            subdirlist = os.listdir(d)
            print('subdirlist:', subdirlist)
            # subdirlist是1类下 所有图片组成的列表
            self.ids += [os.path.join(d, f) for f in subdirlist
                         if os.path.isfile(os.path.join(d, f))]
            print('self.ids:', self.ids)
            try:
                target_name = int(d[-1]) - 1
            except:
                raise TypeError(f'directory name should not be {d}')
            self.targets += [target_name] * len(subdirlist)
        self.means = [0.485, 0.456, 0.406]
        self.stds = [0.229, 0.224, 0.225]
        if val:
            print(f'Creating validation dataset with {len(self.ids)} examples')
        else:
            print(f'Creating training dataset with {len(self.ids)} examples & augmentation none')

    '''
    重载len
    '''

    def __len__(self):
        return len(self.ids)

    @classmethod
    def resize(cls, ori_img, size):
        image = cv2.resize(ori_img, (size, size), interpolation=cv2.INTER_NEAREST)
        image = image / 255
        image = image.transpose((2, 0, 1))
        return image

    @classmethod
    def getsubdir(cls, root_dir):
        return [os.path.join(root_dir, dI) for dI in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, dI))]

    '''
    openCV读取数据
    cv2.imdecode()函数从指定的内存缓存中读取数据，并把数据转换(解码)成图像格式;主要用于从网络传输数据中恢复出图像。
    '''

    @classmethod
    def cv_imread(cls, filepath):
        cv_img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), -1)
        return cv_img

    '''
    重载getitem
    x[i]可是为x.__getitem__(x,i)
    '''

    def __getitem__(self, i):
        file = self.ids[i]
        target = self.targets[i]
        try:
            data = self.cv_imread(file)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        except:
            raise RuntimeError(f'{file} not able to be loaded')

        if self.val:
            data = self.resize(data, self.size)
        else:
            data = self.resize(data, self.size)

        data = tc.from_numpy(data)

        return data, target


class PlainDataset(Dataset):
    def __init__(self, data_dir, size=224):
        self.dir = data_dir
        self.size = size
        assert os.path.exists(self.dir), f'Directory {self.dir} not exist'
        self.ids = [f for f in os.listdir(self.dir)
                    if os.path.isfile(os.path.join(self.dir, f))]
        self.means = [0.485, 0.456, 0.406]
        self.stds = [0.229, 0.224, 0.225]
        print(f'Creating dataset with {len(self.ids)} examples')
    
    def __len__(self):
        return len(self.ids)
    
    '''
    归一化，旋转
    '''
    @classmethod
    def preprocess(cls, ori_img, size):
        image = cv2.resize(ori_img, (size, size), interpolation=cv2.INTER_NEAREST)
        image = image / 255
        image = image.transpose((2, 0, 1))
        return image

    '''
    加载图像
    '''
    @classmethod
    def cv_imread(cls, filepath):
        cv_img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), -1)
        return cv_img
    
    @classmethod
    def crop_image_from_gray(cls, img,tol=7):
        if img.ndim ==2:
            mask = img>tol
            return img[np.ix_(mask.any(1),mask.any(0))]
        elif img.ndim==3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            mask = gray_img>tol
            
            check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
            if (check_shape == 0): # image is too dark so that we crop out everything,
                return img # return original image
            else:
                img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
                img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
                img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
                img = np.stack([img1,img2,img3],axis=-1)
            return img
    
    @classmethod
    def load_ben_color(cls, image, sigmaX=30):
        image = cls.crop_image_from_gray(image)
        image = cv2.resize(image, (512, 512))
        image = cv2.addWeighted (image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
        return image
    
    def __getitem__(self, i):
        file = os.path.join(self.dir, self.ids[i])
        try:
            data = self.cv_imread(file)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            raw = data.copy()
        except:
            raise RuntimeError(f'{file} not able to be loaded')
        
        data = self.load_ben_color(data)
        
        data = self.preprocess(data, self.size)
        data = tc.from_numpy(data)
        raw = tc.from_numpy(raw)
        
        for t, m, s in zip(data, self.means, self.stds):
            t.sub_(m).div_(s)
        return data, raw, self.ids[i]