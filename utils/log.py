# coding: utf-8

# last modify: 2019.9.9

"""
logging 模块简单封装
"""

import logging

class Log(object):

    def __init__(self, filename='./log.txt'):
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

