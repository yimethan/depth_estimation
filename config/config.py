class Config(object):

    batch_size = 32
    epochs = 1
    lr = 1e-2

    gpu = True

    model_name ='my_model.pth'
    log_dir = './outputs/log'
    checkpoint_dir = './outputs/ckp'

    centernet_path = '../centernet-models/best.pth'
    dataset_path = '../dataset/raw_data'

    height = 512
    width = 512

    log_f = 125
    save_f = 1
    scheduler_f = 15

    max_disp = 160

    split = 'eigen_full'