import numpy as np
from torchvision import transforms
from PIL import Image


class ResizeImage():
    def __init__(self, size):
      if isinstance(size, int):
        self.size = (int(size), int(size))
      else:
        self.size = size
    def __call__(self, img):
      th, tw = self.size
      return img.resize((th, tw))


class PlaceCrop(object):
    """Crops the given PIL.Image at the particular index.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))


class SetFlip(object):

    def __init__(self, flip):
        self.flip = flip

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

       
class image_train(object):
    def __init__(self, crop_size, re_size):
        self.crop_size = crop_size
        self.re_size = re_size

    def __call__(self, img, flip, offset_x, offset_y):
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
        transform = transforms.Compose([
            PlaceCrop(self.crop_size, offset_x, offset_y),
            ResizeImage(self.re_size),
            SetFlip(flip),
            transforms.ToTensor(),
            normalize
        ])
        img = transform(img)
        return img



def image_test(crop_size=176,re_size=160):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    return transforms.Compose([
        transforms.CenterCrop(crop_size),
        ResizeImage(re_size),
        transforms.ToTensor(),
        normalize
    ])