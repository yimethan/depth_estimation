class Config(object):

    batch_size = 1
    epochs = 5
    lr = 1e-2

    gpu = True
    num_workers = 0

    model_name = 'my_model.pth'
    log_dir = './outputs/log'

    dataset_path = '../dataset/raw_data'

    gen_newinp_path = './gen_newinp/'
    gen_newpred_path_final = './gen_newpred/final/'
    gen_newpred_path_orig = './gen_newpred/orig/'
    gen_newgt_path = './gen_newgt/'

    height = 128
    width = 256

    detect_height = 640
    detect_width = 640

    full_res_shape = (1242, 375)

    log_f = 125
    save_f = 1
    scheduler_f = 15

    maxdisp = 128

    # last_epoch_path = './outputs/models/weights_epoch0_4000'
    # centernet_path = '../../centernet-re/best.pth'