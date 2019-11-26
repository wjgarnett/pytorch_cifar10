#coding: utf-8

class Config(object):
    use_gpu = True

# hyper parameters
    num_epoch = 400
    batch_size = 256
    learning_rate = 0.01

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

# others
    pth_file = "./checkpoints/net1st.pth"
    log_file = "./log/log.txt"
    log_iter = 50
    visualize = False

#
    resume = True
    best_acc = 0.0
    resume_epoch = 0

CFG = Config()


