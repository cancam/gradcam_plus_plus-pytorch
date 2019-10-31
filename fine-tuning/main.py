from fineTune import fineTune

if __name__ == '__main__':
    
    test_models = True
    test_dataloaders = False
    work_dir = '/home/cancam/imgworkspace/gradcam_plus_plus'

    models = ['resnet', 'vgg', 'densenet']
    num_classes = 80
    batch_size = 64
    num_epochs = 89
    work_dir = '/home/cancam/imgworkspace/gradcam_plus_plus-pytorch'
    if test_models:

        for model in models:
            print("Testing model {}\n".format(model))
            tuner = fineTune(work_dir, model, num_classes, batch_size, num_epochs)
            tuner.init_model(tune_all_params = True, from_scratch = True)
            tuner.get_model()
    
    if test_dataloaders:
        tuner = fineTune(work_dir, models[0], num_classes, batch_size, num_epochs)
        print(tuner.data_transforms)
        dataloader = tuner.init_dataloaders(data_path)
        print(dataloader)
