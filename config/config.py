class Config(object):

    batch_size = 1
    epochs = 5
    lr = 1e-2

    gpu = True
    num_workers = 0

    model_name = 'my_model.pth'
    log_dir = './outputs/log'
    checkpoint_dir = './outputs/ckp'

    centernet_path = '../centernet-re/best.pth'
    dataset_path = '../dataset/raw_data'

    height = 128
    width = 256

    detect_height = 256
    detect_width = 512

    full_res_shape = (1242, 375)

    log_f = 125
    save_f = 1
    scheduler_f = 15

    maxdisp = 128

    split = 'eigen_full'