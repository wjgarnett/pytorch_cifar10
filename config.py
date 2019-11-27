#coding: utf-8

class Config(object):
    use_gpu = True

# hyper parameters
    num_epoch = 2000
    batch_size = 256
    learning_rate = 0.006

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    # mean = (0.5, 0.5, 0.5)
    # std = (0.5, 0.5, 0.5)

# others
    pth_file = "./checkpoints/resnet18.pth"
    log_file = "./log/log.txt"
    log_iter = 50
    visualize = False

#
    resume = True
    best_acc = 0.0
    resume_epoch = 0

CFG = Config()


