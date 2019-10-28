from coco import CocoDataset
import matplotlib.pyplot as plt
import os
import pdb
import pickle

def analysis(dataset_info):
    count = []
    scales = []
    for element in dataset_info:
        count.append(element[0])
        scales.append(element[1])
        pdb.set_trace()

def save_info(dataset_info, PATH, set_name):
    
    folder = 'fine-tune/analysis'
    set_name = set_name[:-4] + '.txt'
    fname = os.path.join(PATH, folder, set_name)
    
    print("Saving dataset info to {}".format(fname))
    with open(fname, "wb") as fp:
        pickle.dump(dataset_info, fp)

def load_info(PATH, set_name):
    folder = 'fine-tune/analysis'
    set_name = set_name[:-4] + '.txt'
    fname = os.path.join(PATH, folder, set_name)

    print("Loading log file from {}".format(fname))
    with open(fname, "rb") as fp:
        dataset_info = pickle.load(fp)
    
    return dataset_info

def do_analysis(dataset_info, PATH, set_name):
    count = []
    scales = []
    folder = 'fine-tune/analysis'
    plt_name = os.path.join(PATH, folder, set_name[:-4])
    for element in dataset_info:
        count.append(element[0])
        scales.append(element[1])
    scales_ = [val for scale in scales for val in scale]
    
    plt.subplot(2, 1, 1)
    plt.hist(count)
    plt.title("Number of parts per image")
    plt.ylabel("Count")
    
    plt.subplot(2, 1, 2)
    plt.hist(scales_)
    plt.title("Scales for each part")
    plt.xlabel("Bins")
    plt.ylabel("Count")
    
    plt.tight_layout()
    plt.savefig(plt_name+'.png')

def main(PATH, set_name, extract, analysis):
    
    if extract:
        coco_ds = CocoDataset(PATH, set_name)
        dataset_info = coco_ds.process_images()
        save_info(dataset_info, PATH, set_name)
    
    if analysis:
        dataset_info = load_info(PATH, set_name)
        do_analysis(dataset_info, PATH, set_name)
    

if __name__ == '__main__':
    PATH = '/home/cancam/workspace/gradcam_plus_plus-pytorch/data/coco'
    set_name = 'val2017'
    extract = True
    analysis = False
    main(PATH, set_name, extract, analysis)
