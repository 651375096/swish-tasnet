import lvm_read as lvm
import numpy.matlib
import numpy as np
import matplotlib.pyplot as plt
import soundfile
import re
import os #os模块中包含很多操作文件和目录的函数

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
        #soundfile.write(path, data, 44100, format='wav', subtype='PCM_16')
        soundfile.write(path, data, 8000, format='wav', subtype='PCM_16')
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



def isNumber(str):
    '''判断输入字符串是否为数字

    :param str: 输入的字符串
    :return flag: 返回判断结果
    '''
    value = re.compile(r'^(\-|\+)?\d+(\.\d+)?$')
    result = value.match(str)

    if result:
        flag = True
    else:
        flag = False

    return flag

def handleBinarization(data,input_standard,deviation,way=0):
    '''进行二值化操作

    :param data: 需要进行二值化的矩阵
    :param standard_str: "range" 极差   "average"平均数  默认standard=0  "数字" 标准就是该数字值
    :param deviation: 偏差值
    :param way: 进行二值化的方式
    :return:bin_data 二值化后的矩阵
    '''
    bin_data=np.copy(data)
    if isNumber(input_standard):
        standard=float(input_standard)
    elif  input_standard=="range":
        standard = (np.max(bin_data) + np.min(bin_data)) / 2  # 极差作为标准
    elif input_standard=="average":
        standard = np.mean(bin_data) #平均数作为标准
    elif input_standard=="min":
        standard = np.min(bin_data)  # 最小值作为标准
    #print("standard:"+str(standard))
    if( way==1 ):
        bin_data[np.where(bin_data > (standard + deviation))] = 0  # 二值化 大于标准为0，小于标准为1
        bin_data[np.where(bin_data != 0)] = 1
    else:
        bin_data[np.where(bin_data < (standard - deviation))] = 0  # 二值化 小于标准为0，大于标准为1
        bin_data[np.where(bin_data != 0)] = 1

    return bin_data

def handleAbsolute(data):
    '''该函数用来进行取绝对值操作

    :param data: 需要处理的数据
    :return: 进行绝对值处理后的数据
    '''
    abs_data = abs(data)
    return abs_data

def handleDifference(data,flag=1):
    '''进行差分操作

    :param data: 需要进行差分的矩阵
    :param flag: 是否需要补充差分缺少的元素
    :return: 返回差分矩阵
    '''
    dif_data=np.diff(data, axis=0)
    #print(dif_data)
    #差分是用后一个元素减去前一个元素，导致元素减少，在末尾补充一个尾元素0
    #index = len(dif_data)-1
    #append_data = dif_data[index]
    if(flag==1):
        append_data = [0]   #注意这里dif_data是二维数组，所以加上的0要是一维数组[0]
        dif_data_append = np.append(dif_data, [append_data], axis=0)
    else:
        dif_data_append = dif_data
    #print(dif_data_append)
    return dif_data_append


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
    plt.savefig("./picture/" + pic_name + ".jpg")
    plt.show()

def getPosition(data,value):
    '''该函数用来获取data数据中值value所在的位置

    :param data: 搜索的数据
    :param value: 需要获取的值
    :return:位置矩阵
    '''
    position_array=np.where(data==value)
    position=position_array[0]
    return position


def replacePoints(data,position,point_num,aim_value,deviation=5):
    '''

    :param data: 需要处理的数据
    :param position: 需要替换的位置
    :param point_num: 替换的点数
    :param aim_value:替换的目标值
    :param deviation:偏差值
    :return: 替换后的数据
    '''

    replace_data = np.copy(data)
    for i in position:
        for j in range(point_num):
            index=i-deviation + j + 1 #为了避免位置误差向左偏移5个点
            if(index<0):
                index = 0
            #if( (index>=len(replace_data)) or  (replace_data[index]==aim_value)):
            if (index >= len(replace_data)):
                break
            replace_data[index]=aim_value

    return replace_data

def getMarsk(data,input_standard,deviation,replace_point,point_num,aim_value):
    '''改函数用来获取相应数据的掩码

    :param data: 需要获取掩码的数据
    :param input_standard: 进行二值化的标准
    :param deviation: 进行二值化运行的误差
    :param replace_point: 进行替换时定位点的值
    :param point_num: 替换点的数量
    :param aim_value: 替换值
    :return: 返回掩码文件
    '''

    #将数据取绝对值
    abs_data=handleAbsolute(data)
    #drawPicture("1", abs_data)
    # 对数据进行二值化操作
    bin_data=handleBinarization(abs_data, input_standard, deviation)
    #drawPicture("2",bin_data)
    #对数据进行差分
    diff_data=handleDifference(bin_data)
    #drawPicture("2", diff_data)
    # 将数据取绝对值
    abs_diff_data = handleAbsolute(diff_data)
    #drawPicture("2", abs_diff_data)
    #取数据中1所在的位置
    position=getPosition(abs_diff_data, replace_point)
    #print(position)
    #替换点，获取掩码文件
    mask_data=replacePoints(abs_diff_data, position, point_num, aim_value)
    #drawPicture("2", mask_data)
    return mask_data



def getSignalInfornamtion(frame_data,bin_deviation=0):
    '''该函数用来获取信号的帧信息

    :param frame_data: 该信号的帧同步信号
    :param bin_deviation: 二值化允许的误差 默认值为 0.1
    :return:信号的帧信息  [{'start': 532368, 'end': 4667857, 'len': 4135489}，...]
    '''

    # 根据帧信号标识的方向，确定二值化方式
    max_value = np.max(frame_data)
    min_value = np.min(frame_data)
    average_value = np.mean(frame_data)
    if (abs(max_value - average_value) >= abs(min_value - average_value)):
        way = 0
    else:
        way = 1
    #二值化 标准使用极差的形式  二值化方式为1
    bin_data = handleBinarization(frame_data, "range", bin_deviation , way)
    temp_position = np.where(bin_data == 1)

    '''
(array([  659689,   659691,   659692,   659693,   659694,  4826186,
        4826187,  4826189,  4826190,  4826191,  8992683,  8992684,
        8992685,  8992686,  8992687,  8992688, 13159179, 13159180,
       13159181, 13159183, 13159184, 17325676, 17325677, 17325678,
       17325679, 17325680, 21492174, 21492175, 21492176, 21492178,
       25658670, 25658671, 25658673, 29825167, 29825168, 29825169,
       29825170, 29825171, 33991661, 33991662, 33991663, 33991664,
       33991665, 33991666, 33991667, 38158160, 38158161, 38158162,
       38158163, 38158164, 38158165, 42324657, 42324658, 42324659,
       42324660, 46491154, 46491155, 46491156, 46491157, 46491158],
      dtype=int64), array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int64))

         '''
    #获取位置的差值数组
    diff_temp_position = handleDifference(temp_position[0],0)
    #取差值的平均值
    standard = np.mean(diff_temp_position)
    #根据差值的平均值获取信号帧位置信息
    position = findPartPosition(temp_position[0], standard)

    return position


def findPartPosition(data,standard):
    '''该函数用来获取数组中，两个相邻元素差值大于standard的位置

    :param data: 需要处理的数组
    :param standard: 处理标准
    :return: 位置信息 [{'start': 532368, 'end': 4667857, 'len': 4135489}，...]
    '''
    i=0
    j=1
    list_position=[]
    length= len(data)
    while(j< length):
        if( data[j]-data[i] > standard):
            dic_position = {'start': data[i], 'end': data[j], 'len': data[j] - data[i]}
            list_position.append(dic_position)

        i=i+1
        j=j+1

    return list_position

def findMaxLength(*lens):
    '''该函数用来获取一组长度中的最大长度

    :param lens: 一组长度值
    :return max_len: 最大长度
    '''

    max_len=np.max(lens)

    return max_len


def var_name(var,all_var=locals()):
    '''该函数用来获取变量名

    :param var: 需要获取变量名的变量
    :param all_var: 所以变量
    :return: 返回变量的变量名
    '''
    return [var_name for var_name in all_var if all_var[var_name] is var][0]

def reverseMatrix(org_matrix):
    '''该函数用来将由0-1构成的矩阵取反

    :param org_matrix: 需要取反的矩阵
    :return: 取反后的矩阵
    '''
    #构造一维的全1矩阵
    one_matrix = np.ones(len(org_matrix))
    #将全1矩阵维度调整成和org_matrix相同
    one_matrix.shape=org_matrix.shape
    #相减实现取反
    reverse_matrix = one_matrix - org_matrix
    return reverse_matrix


def fillBlanks(position,data,distance,deviation=200):
    '''该函数用来补充真实噪声信号中别扣除的点

    :param position: 位置的掩码，替换的位置处1，不需要替换为0
    :param data: 需要处理的数据
    :param distance: 替换的距离
    :param deviation: 处理阴影的误差
    :return: 处理的结果
    '''

    fill_data = np.copy(data)
    for i in range(len(position)):
        if position[i]==1:
            index = i - distance  # 0 1 2  3 4
            if (index >= 0):
                fill_data[i] = fill_data[index]
            if ((i+1)<len(position)) and (position[i+1] == 0):# 用来消除阴影
                for j in range(deviation):
                    if (i+1+j) >= len(fill_data):
                        break
                    fill_data[i+1+j] = fill_data[index+j]

    return fill_data

def sliceData(data,ways,fileDir,fileName,slice_size):
    '''该函数用来划分数据集

    :param data: 需要划分的数据
    :param ways: 存储方式
    :param fileDir: 存放的路径
    :param fileName: 文件名
    :param slice_size: 划分尺寸
    :return: 返回划分结果 True or False
    '''

    begin_position = 0
    end_position = begin_position + slice_size
    result = False
    while(True):
        if(end_position >= len(data)):
            end_position = len(data)
            result=True
            break

        file_name = fileName +'_' +str(begin_position) +'_'+ str(end_position)
        writeData(data[begin_position:end_position], ways, fileDir, file_name)

        begin_position = end_position
        end_position = begin_position + slice_size

    return result



def getLineLength(line_data):

    positions = getSignalInfornamtion(line_data)
    sum=0
    for pos in positions:
        sum= sum + pos["len"]

    avg_len = round(sum / len(positions))

    return avg_len


#读取lvm文件
# jmu_frame=readLvm("./data/1024_768_JMU/1024_768_60_JMU_250MHz.lvm")
# black_frame=readLvm("./data/1024_768_JMU/1024_768_60_black_250MHz.lvm")
# blue_frame=readLvm("./data/1024_768_JMU/1024_768_60_B_250MHz.lvm")

# jmu_frame=readLvm("./data/OriginalElectData/1024_768_JMU/1024_768_60_JMU_250MHz.lvm")
# black_frame=readLvm("./data/OriginalElectData/1024_768_JMU/1024_768_60_black_250MHz.lvm")
# blue_frame=readLvm("./data/OriginalElectData/1024_768_JMU/1024_768_60_B_250MHz.lvm")
line_frame=readLvm("./data/OriginalElectData/1024_768_JMU/1024_768_60_ZHENHANG_250MHz.lvm")


# jmu_frame = readLvm("./data/OriginalElectData/848_480_JMU/848_480_60_JMU_250MHz.lvm")
# black_frame = readLvm("./data/OriginalElectData/848_480_JMU/848_480_60_black_250MHz.lvm")
# blue_frame = readLvm("./data/OriginalElectData/848_480_JMU/848_480_60_B_250MHz.lvm")

# jmu_frame = readLvm("./data/OriginalElectData/800_600_JMU/800_600_60_JMU_250MHz.lvm")
# black_frame = readLvm("./data/OriginalElectData/800_600_JMU/800_600_60_black_250MHz.lvm")
# blue_frame = readLvm("./data/OriginalElectData/800_600_JMU/800_600_60_B_250MHz.lvm")

# jmu_frame = readLvm("./data/OriginalElectData/640_480_JMU/640_480_60_JMU_250MHz.lvm")
# black_frame = readLvm("./data/OriginalElectData/640_480_JMU/640_480_60_black_250MHz.lvm")
# blue_frame = readLvm("./data/OriginalElectData/640_480_JMU/640_480_60_B_250MHz.lvm")


# drawPicture("jmu_frame_1024_768",jmu_frame["signal"],jmu_frame["frame"])
# drawPicture("black_frame_1024_768",black_frame["signal"],black_frame["frame"])
# drawPicture("blue_frame_1024_768",blue_frame["signal"],blue_frame["frame"])
#drawPicture("line_frame_1024_768_44100",line_frame["signal"][0:44100],line_frame["frame"][0:44100])

# positions = getSignalInfornamtion(line_frame["signal"])
# drawPicture("line_1024_768",
#             line_frame["signal"][positions[0]["start"]:positions[0]["end"]],
#             line_frame["signal"][positions[1]["start"]:positions[1]["end"]],
#             line_frame["signal"][positions[2]["start"]:positions[2]["end"]],
#             line_frame["signal"][positions[3]["start"]:positions[3]["end"]]
#             )
print(getLineLength(line_frame["signal"])*70)

