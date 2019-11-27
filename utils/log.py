# coding: utf-8

# last modify: 2019.9.9

"""
logging 模块简单封装
    用法:
        from utils.log import logger as logging
        logging.info("----")
"""
import os

log_dir = './log'
log_file = './log/log.txt'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

import logging

class Log(object):

    def __init__(self, filename=log_file):
        '''
        默认将日志同时输出到控制台及文件
        '''
        logging.basicConfig(
            level=logging.DEBUG,    #调式时设置为DEBUG level, 运行时用INFO level即可
            format='%(asctime)s - %(filename)s - %(funcName)s - %(lineno)d - %(levelname)s - %(message)s',
            filename=filename
        )

        logger = logging.getLogger('')

        console = logging.StreamHandler() #控制台输出
        console.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(filename)s - %(funcName)s - %(lineno)d - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        logger.addHandler(console)  #addHandler要setLevel setFormatter才行

        self.logger = logger




# def test():
#     logging.info('test info')
#     logging.warning('test debug')
#
# if __name__ == '__main__':
#
#     # logging模块配置
#     Log()
#
#     test()

if 'logger' not in locals().keys():
    logger = Log().logger

