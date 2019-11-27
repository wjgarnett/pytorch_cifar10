#coding utf-8
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import StepLR
from torchvision import models

from dataloader.cifar10_loader import CIFAR10Loader
from models.net1st import Net1st
from models.resnet import ResNet18

from config import CFG
import logging
from utils.log import Log

def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
make_if_not_exist("./checkpoints")
make_if_not_exist("./log")
# if not os.path.exists(CFG.log_file):
#     os.makedirs(CFG.log_file)
Log(CFG.log_file)


def train(model, device, train_loader, optimizer, criterion, epoch, log_iter=1000):
    model.train() #加上该句标明是训练阶段，BN和dropout保留。model.eval()作用类似
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader): #data, target比inputs/labels更通用
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data) #forward
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss
            # log打印
        if batch_idx % log_iter == log_iter-1:
            logging.info('Train Epoch: {} [{}/{}] loss: {}'.format(epoch, batch_idx*len(data), len(train_loader.dataset),
                                                                   running_loss/log_iter))
            running_loss = 0.0
    # writer.add_scalar("training loss", running_loss / log_iter, global_step=epoch)
    #writer.close()



def test(model, device, test_loader, criterion, epoch):
    global best_acc
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)

    logging.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    # writer.add_scalar("test loss", test_loss, global_step=epoch)
    #writer.close()

    #保存当前测试集准确率最高的模型
    acc = 100.0*correct / len(test_loader.dataset)
    if acc > CFG.best_acc:
        logging.info("Saving current best model ...")
        logging.info("acc: {}, best_acc: {}".format(acc, CFG.best_acc))
        # print("model:\n", model)
        state = {
            "model": model.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        if device == 'cuda' and torch.cuda.device_count() > 1:
            state['model'] = model.module.state_dict() #多gpu训练时对网络结构添加了module进行了封装,需要解封装保存
        torch.save(state, CFG.pth_file)
        CFG.best_acc = acc


def main():
    #---数据加载+数据增强---
    train_set = CIFAR10Loader(root='./Data', train=True, transform=True)
    test_set = CIFAR10Loader(root='./Data', train=False, transform=False)
    # print("trainset: ", len(trainset))
    # print("testset: ", len(testset))
    train_loader = DataLoader(train_set, batch_size=CFG.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=CFG.batch_size, shuffle=False, num_workers=2)

    #---网络模型定义及加载---
    model = ResNet18()
    # model = models.resnet18(pretrained=True)
    # model.fc = nn.Linear(in_features=512, out_features=10)
    # params = list(model.parameters())
    # print(model)
    #
    # for para in params[:-1]:
    #     para.requires_grad = False
    # print(len(params))
    # print(params)
    # model = Net1st()
    if CFG.resume:
        # pth_file = os.path.join(CFG.ckpt_dir, CFG.file_name)
        if os.path.exists(CFG.pth_file):
            logging.info("resume mode from {} ...".format(CFG.pth_file))
            ckpt = torch.load(CFG.pth_file)
            checkpoint = torch.load(CFG.pth_file)
            model.load_state_dict(checkpoint['model'])
            CFG.best_acc = checkpoint['acc']  # 对全局变量赋值
            CFG.resume_epoch = checkpoint['epoch'] + 1
            logging.info("resume from epoch {}, best acc is {} ...".format(CFG.resume_epoch, CFG.best_acc))

    #---训练环境配置---
    if CFG.use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"

    if device == 'cuda' and torch.cuda.device_count() > 1:
        logging.info("train on {} gpus ...".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
        cudnn.benchmark = True
    model = model.to(device)  # to device

    #---优化器损失函数定义+学习率调整策略设置---
    optimizer = optim.SGD(model.parameters(), lr=CFG.learning_rate, momentum=0.9)
    # optimizer = optim.Adadelta(model.parameters(), lr=CFG.learning_rate)
    # optimizer = optim.Adam(model.parameters(), lr=CFG.learning_rate)
    criterion = nn.CrossEntropyLoss()  # 损失函数
    # scheduler = ExponentialLR(optimizer, 0.9) #学习率调整策略
    scheduler = StepLR(optimizer, 10, gamma=0.95)

    #---train---
    for epoch in range(CFG.resume_epoch, CFG.resume_epoch+CFG.num_epoch):
        logging.info("epoch: {}, learning rate: {}".format(epoch, scheduler.get_lr()))
        train(model, device, train_loader, optimizer, criterion, epoch, log_iter=CFG.log_iter)
        test(model, device, test_loader, criterion, epoch)
        scheduler.step()
        logging.info('best_acc: {}'.format(CFG.best_acc))

    logging.info('Finished Training !!!')

if __name__ == '__main__':
    main()

