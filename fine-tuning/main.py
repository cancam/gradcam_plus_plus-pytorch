from fineTune import fineTune

if __name__ == '__main__':

    models = ['resnet', 'vgg', 'densenet']
    num_classes = 80
    batch_size = 64
    num_epochs = 89
    work_dir = '/truba/home/bcam/imgworkspace/gradcam_plus_plus-pytorch'
    tuner = fineTune(work_dir, models[2], num_classes, batch_size)
    tuner.init_model(tune_all_params = False, from_scratch = False)
    tuner.train_model()
