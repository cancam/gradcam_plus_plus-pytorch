from coco import CocoDataset
import os
import pdb


def main(annotations):
    coco_ds = CocoDataset(annotations)
    pdb.set_trace()

if __name__ == '__main__':
    PATH = '/home/cancam/workspace/gradcam_plus_plus-pytorch/data/coco/annotations'
    ann_file = 'instances_train2017.json'
    annotations = os.path.join(PATH, ann_file)
    main(annotations)
