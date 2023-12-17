import lvm_read as lvm
import numpy as np
import matplotlib.pyplot as plt
import soundfile


def write_wav(data, path):  # 将信号保存到指定路径
    soundfile.write(path, data, 44100, format='wav', subtype='PCM_16')


mix = lvm.read("E:\\Files\\code\\pretreatment\\1024_768_JMU\\1024_768_60_JMU_250MHz.lvm")
print("mix\n")
#black = lvm.read("E:\\Files\\code\\pretreatment\\1024_768_JMU\\1024_768_60_black_250MHz.lvm")
print("black\n")
#blue = lvm.read("E:\\Files\\code\\pretreatment\\1024_768_JMU\\1024_768_60_B_250MHz.lvm")
print("blue\n")

'''
总量：45831472
训练集：0-20000000
验证集：20000000-40000000
测试集：40000000-45830000
'''

mix = mix[0]
mix = mix['data']
print(mix)
mix = mix[:,0:1]    #JMU第一路混合信号
#mix = mix[40000000:45830000]  #取到第一帧同步
#write_wav(mix, 'E:\\Files\\code\\pretreatment\\electromagnetism\\tt\\mix\\h1.wav')
print(mix)
'''
write_wav(mix, 'E:\\Files\\code\\pretreatment\\electromagnetism\\JMU.wav')

black = black[0]
black = black['data']
black = black[:,0:1]    #JMU第一路混合信号
#black = black[40000000:45830000]  #取到第一帧同步
#write_wav(black, 'E:\\Files\\code\\pretreatment\\electromagnetism\\tt\\s1\\h1.wav')
write_wav(black, 'E:\\Files\\code\\pretreatment\\electromagnetism\\black.wav')

blue = blue[0]
blue = blue['data']
blue = blue[:,0:1]    #JMU第一路混合信号
#blue = blue[40000000:45830000]  #取到第一帧同步
#write_wav(blue, 'E:\\Files\\code\\pretreatment\\electromagnetism\\tt\\s2\\h1.wav')
write_wav(blue, 'E:\\Files\\code\\pretreatment\\electromagnetism\\blue.wav')
'''