class Config(object):

    batch_size = 32
    epochs = 1

    model_name = 'my_model.pth'
    
    root = '../dataset/raw_data'

    resize_size = [512, 512]
    height = 512
    width = 512

    optimizer = 'AdamW'
    lr = 1e-2
    AMSGRAD = True

    gpu = True
    eval = True

    log_dir = './results/log'
    checkpoint_dir = './results/ckp'

    centernet_path = '../centernet-models/best.pth'

    num_workers = 12
    split = 'eigen_zhou'
    use_stereo = True

    log_frequency = 125 # num of batches between each tensorboard log 250
    save_frequency = 1 # num of epochs between each save 1

    pred_depth_scale_factor = 1
    eval_split = 'eigen'

    max_disp = 160 # 100?

    png = True
    scheduler_step_size = 15
    max_iter = None
    steps = None

    use_stereo = True
    frame_ids = [0, -1, 1]
    scales = [0, 1, 2, 3]

    no_cuda = False
    load_weights_folder = None