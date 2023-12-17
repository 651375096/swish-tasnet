import os
import soundfile
import numpy as np
#auto add model name

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
        soundfile.write(path, data, 44100, format='wav', subtype='PCM_16')
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

np.set_printoptions(suppress=True)

#estimate    original
estimate_s1_save_dir ="../exp/tmp/out/signal/original/mix/"
#estimate_s1_save_dir ="../voice_data/wav/tt/s1/"
file_name = os.listdir(estimate_s1_save_dir)
sort_file_name = sorted(file_name,key = lambda i: int(i.split(".")[0]))

temp_data=[]
for name in sort_file_name:
    temp = np.loadtxt(estimate_s1_save_dir + name)
    temp_data = np.append(temp_data, temp)
    #temp_data.shape(temp_data_len,1)
    #print(temp_data)

data=temp_data.reshape(len(temp_data),1)
ways="lvm"
fileDir="./data/"
fileName = "original_mix"
writeData(data, ways,fileDir,fileName)