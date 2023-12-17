import lvm_read as lvm
import numpy.matlib
import numpy as np
import matplotlib.pyplot as plt
import soundfile
import re
import os #os模块中包含很多操作文件和目录的函数
import shutil #移动文件夹命令

def readLvm(dir):
    '''读取电磁信号的lvm文件

    :param dir: 读取路径
    :return:返回一个读取的lvm字典 关键字：signal对于包含的信号  关键字frame对于帧同步信号
    '''
    lvm_file=lvm.read(dir)
    lvm_file = lvm_file[0]
    lvm_data = lvm_file['data']
    lvm_data = np.array(lvm_data)  # 装入数组
    signal_frame = {'signal':lvm_data[:,0:1], 'frame':lvm_data[:,1:2]}
    return signal_frame

def writeData(data, ways,fileDir,fileName,lvm_head_path="./data/lvm_head.lvm"):  # 将信号保存到指定路径
    '''该函数用来写入数据

    :param data:需要写入的数据
    :param ways:写入方式：txt 、 wav、lvm、lvm_head
    :param fileDir:需要写入的路径
    :param fileName:需要写入的文件名
    :param lvm_head_path:  当选择写入的文件为lvm文件时，需要填入lvm头文件地址，默认为"./data/lvm_head.lvm"
    :return result: 返回是否成功
    '''
    result = True

    #如果文件件不存在创建文件夹
    if not os.path.exists(fileDir):
        os.makedirs(fileDir)

    temp_path = fileDir + fileName

    if(ways=='wav'):
        path = temp_path +'.'+ways
        #soundfile.write(path, data, 8000, format='wav', subtype='PCM_16')
        soundfile.write(path, data,44100 , format='wav', subtype='PCM_16')
    elif(ways=='txt'):
        path = temp_path + '.' + ways
        np.savetxt(path, data ,fmt='%.05f')
    elif (ways == 'lvm'):
        #生成lvm数据文件
        path = temp_path + '.' + ways
        np.savetxt(path, data, fmt='%.05f')
        #读取lvm头文件
        lvm_head = np.array([readHead(lvm_head_path)], dtype=object)
        #写入lvm头文件
        path = temp_path + 'Head.' + 'lvm'
        np.savetxt(path, lvm_head, fmt='%s')
        #将数据追加到头文件后   生成lvm头部+数据文件
        with open(path, "ab") as f:
            np.savetxt(f, data, fmt='%.05f')
    else:
        result = False

    return result

def readHead(filePath):
    '''该函数用来读取文件，并以字符串形式输出

    :param filePath: 读取文件路径
    :return: 返回文件内容的字符串
    '''

    # 打开aimFilePath文件，如果没有则创建,文件也可以是其他类型的格式
    file = open(filePath)
    file_str=''
    # 遍历单个文件，读取行数
    for line in file:
        file_str=file_str+line

    # 关闭文件
    file.close()

    return file_str


def drawPicture(pic_name,*data):
    '''该函数用来绘制图片

    :param pic_name: 图片保存的文件名
    :param data: 要绘制的数据
    :return: none
    '''
    num=len(data)
    #print(num)
    for i in range(num):
        y = data[i]
        plt.subplot(num, 1, i+1)  # 第num个图画在一个两行一列分割图的第i幅位置
        plt.plot(y)
    # plt.savefig("./picture/" + pic_name + ".jpg")
    plt.show()

# jmu_frame = readLvm("./data/OriginalElectData/848_480_JMU/848_480_60_JMU_250MHz.lvm")
# black_frame = readLvm("./data/OriginalElectData/848_480_JMU/848_480_60_black_250MHz.lvm")
# blue_frame = readLvm("./data/OriginalElectData/848_480_JMU/848_480_60_B_250MHz.lvm")

# jmu_frame = readLvm("./data/OriginalElectData/800_600_JMU/800_600_60_JMU_250MHz.lvm")
# black_frame = readLvm("./data/OriginalElectData/800_600_JMU/800_600_60_black_250MHz.lvm")
# blue_frame = readLvm("./data/OriginalElectData/800_600_JMU/800_600_60_B_250MHz.lvm")

# jmu_frame = readLvm("./data/OriginalElectData/640_480_JMU/640_480_60_JMU_250MHz.lvm")
# black_frame = readLvm("./data/OriginalElectData/640_480_JMU/640_480_60_black_250MHz.lvm")
# blue_frame = readLvm("./data/OriginalElectData/640_480_JMU/640_480_60_B_250MHz.lvm")


jmu_frame=readLvm("./data/OriginalElectData/1024_768_JMU/1024_768_60_JMU_250MHz.lvm")
# black_frame=readLvm("./data/OriginalElectData/1024_768_JMU/1024_768_60_black_250MHz.lvm")
# blue_frame=readLvm("./data/OriginalElectData/1024_768_JMU/1024_768_60_B_250MHz.lvm")
# line_frame=readLvm("./data/OriginalElectData/1024_768_JMU/1024_768_60_ZHENHANG_250MHz.lvm")




# drawPicture("jmu_frame_1024_768",jmu_frame["signal"],jmu_frame["frame"])
# drawPicture("black_frame_1024_768",black_frame["signal"],black_frame["frame"])
# drawPicture("blue_frame_1024_768",blue_frame["signal"],blue_frame["frame"])
# drawPicture("line_frame_1024_768_44100",line_frame["signal"],line_frame["frame"])


writeData(jmu_frame["signal"][0:5000000], "lvm","./data/","test",lvm_head_path="./data/lvm_head.lvm")