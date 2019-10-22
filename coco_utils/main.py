from coco import CocoDataset
import os
import pdb


def main(PATH, set_name):
    coco_ds = CocoDataset(PATH, set_name)
    coco_ds.preprocess_images()

if __name__ == '__main__':
    PATH = '/home/cancam/workspace/gradcam_plus_plus-pytorch/data/coco'
    set_name = 'train2017'
    main(PATH, set_name)
