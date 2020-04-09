from fineTune import fineTune

if __name__ == '__main__':

    models = ['resnet', 'vgg', 'densenet']
    num_classes = 80
    batch_size = 64
    num_epochs = 89
    input_size = (224,224)
    work_dir = '/home/udemirezen/baris/imgworkspace/gradcam_plus_plus-pytorch'
    tuner = fineTune(work_dir, models[0], num_classes, batch_size, input_size)
    tuner.init_model(tune_all_params = True, from_scratch = True)
    tuner.train_model(num_epochs=num_epochs)
