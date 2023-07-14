class Config(object):
    # system sets
    USE_CUDA=True
    RESUME=False

    #  Path
    source_path="."
    data_path= source_path + "/data"
    save_dir = source_path + '/checkpoints/mobilenetv2'

    path_checkpoint = save_dir+"/model.th"  # 断点路径

    dataset = 'gesture'
    backbone = 'mobilenetv2'
    classify = 'softmax'
 


    train_batch_size = 8	  # batch size
    test_batch_size = 8

    optimizer = 'sgd'

    num_workers = 8
    print_freq = 20
    save_every = 2


    max_epoch = 20
    lr = 1e-3

