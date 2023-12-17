import lvm_read as lvm
import numpy.matlib
import numpy as np
import matplotlib.pyplot as plt
import soundfile
import re
import os #os模块中包含很多操作文件和目录的函数
import shutil #移动文件夹命令


def dealWavToTxt(wav_path,txt_path):
    '''

    :param wav_path:需要转换的wav文件
    :param txt_path:保存txt的路径
    :return:
    '''
    data = soundfile.read(wav_path)[0]
    np.savetxt(txt_path, data, fmt='%.05f')

path = "./data/1.wav"
#soundfile.write(path, data, 44100, format='wav', subtype='PCM_16')

# data= np.array([[0.0],[0.0],[0.00006]])
#
# soundfile.write(path, data, 8000, format='wav', subtype='PCM_16')

# data = soundfile.read(path)[0]
# np.savetxt("./data/1.txt", data ,fmt='%.05f')
# print(data)
dealWavToTxt(path,"./data/1.txt")
