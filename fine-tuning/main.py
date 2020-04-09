from fineTune import fineTune

if __name__ == '__main__':

    models = ['resnet', 'vgg', 'densenet']
    num_classes = 80
    batch_size = 64
    num_epochs = 89
    work_dir = '/home/udemirezen/baris/imgworkspace/gradcam_plus_plus-pytorch'
    tuner = fineTune(work_dir, models[0], num_classes, batch_size)
    tuner.init_model(tune_all_params = False, from_scratch = True)
    tuner.train_model()
