import torchvision.transforms.functional as F
from torchvision import transforms
import numpy as np
import numbers
import pdb

def get_padding(image, divisor):    
    pad_h = int(np.ceil(image.size[0] / divisor)) * divisor
    pad_w = int(np.ceil(image.size[1] / divisor)) * divisor
    return pad_h, pad_w

class NewPad(object):
    def __init__(self, divisor, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode
        self.divisor = divisor 
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return F.pad(img, get_padding(img, self.divisor), self.fill, self.padding_mode)
    
    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.fill, self.padding_mode)
