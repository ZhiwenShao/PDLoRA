import numpy as np
import random
from PIL import Image


def make_dataset(image_list, label):
    len_ = len(image_list)
    images = [(image_list[i].strip(), label[i, :]) for i in range(len_)]
    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # Image value: [0,1]
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def default_loader(path):
    return pil_loader(path)


class ImageList_au(object):

    def __init__(self, path, crop_size=176, phase='train', transform=None, target_transform=None,
                 loader=default_loader):

        image_list = open(path + '_path.txt').readlines()
        labels = np.loadtxt(path + '_AUoccur.txt', dtype=int)

        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise (RuntimeError('Found 0 images in subfolders of: ' + path + '\n'))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.crop_size = crop_size
        self.phase = phase

    def __getitem__(self, index):
        '''
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        '''
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.phase == 'train':
            w, h = img.size
            offset_y = random.randint(0, h - self.crop_size)
            offset_x = random.randint(0, w - self.crop_size)

            flip = random.randint(0, 1)

            if self.transform is not None:
                img = self.transform(img, flip, offset_x, offset_y)
        else:
            if self.transform is not None:
                img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)