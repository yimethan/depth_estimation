class Config(object):

    batch_size = 1
    epochs = 1
    lr = 1e-2

    gpu = True
    num_workers = 0

    model_name = 'my_model.pth'
    log_dir = './outputs/log'
    checkpoint_dir = './outputs/ckp'

    centernet_path = '../centernet-models/best.pth'
    dataset_path = '../dataset/raw_data'

    height = 256
    width = 512

    full_res_shape = (1242, 375)

    log_f = 125
    save_f = 1
    scheduler_f = 15

    maxdisp = 128

    split = 'eigen_full'