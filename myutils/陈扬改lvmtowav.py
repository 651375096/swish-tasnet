import lvm_read as lvm
import numpy.matlib
import numpy as np
import matplotlib.pyplot as plt
import soundfile
import re
import os   #os模块中包含很多操作文件和目录的函数`
import shutil #移动文件夹命令
fbl = "alphot"
hang_cs=35
zhen_savepath="./data/"+(fbl)+"_"+str(hang_cs)+"/"
save_file = "./data/"+fbl+"_"+str(hang_cs)+"train"+"/"
# mix_path="./data/every_signal/ppt4_中文笔画/1024_768/20220401_1024-768zhen+fushe+ppt4.lvm"
# blue_path="./data/every_signal/ppt4_中文笔画/1024_768/20220401_1024-768zhen+b+ppt4.lvm"
# hang_path="./data/every_signal/ppt4_中文笔画/1024_768/20220401_1024-768zhen+hang+ppt4.lvm"

# mix_path="./16data/1024_768_60_jmu_250MHz.lvm"
# blue_path="./16data/1024_768_60_B_250MHz.lvm"
# hang_path="./16data/1024_768_60_ZHENHANG_250MHz.lvm"
#
mix_path ="./data/1024_768_JMU/1024_768_60_JMU_250MHz.lvm"
blue_path = "./data/1024_768_JMU/1024_768_60_B_250MHz.lvm"
hang_path = "./data/1024_768_JMU/1024_768_60_ZHENHANG_250MHz.lvm"

# mix_path="./data/1024_768_Alphabet/1024_768_60_Alphabet_250MHz.lvm"
# blue_path="./data/1024_768_Alphabet/1024_768_60_Blue_250MHz.lvm"
# hang_path="./data/1024_768_Alphabet/1024_768_60_Line_250MHz.lvm"

def readLvm(dir):
    '''读取电磁信号的lvm文件:param dir: 读取路径
    :return:返回一个读取的lvm字典 关键字：signal对于包含的信号  关键字frame对于帧同步信号  '''
    lvm_file=lvm.read(dir)
    lvm_file = lvm_file[0]
    lvm_data = lvm_file['data']
    lvm_data = np.array(lvm_data)  # 装入数组
    signal_frame = {'signal':lvm_data[:,0:1], 'frame':lvm_data[:,1:2]}
    return signal_frame
 # 读取lvm文件  blue_black
# jmu_frame=readLvm("./16data/1024_768_60_jmu_250MHz.lvm")
# blue_frame=readLvm("./16data/1024_768_60_B_250MHz.lvm")
# line_frame = readLvm("./16data/1024_768_60_ZHENHANG_250MHz.lvm")

# jmu_frame=readLvm("./data/every_signal/ppt2_字母表/1024_768/20220401_1024-768zhen+fushe+ppt2.lvm")
# blue_frame=readLvm("./data/every_signal/ppt2_字母表/1024_768/20220401_1024-768zhen+b+ppt2.lvm")
# line_frame = readLvm("./data/every_signal/ppt2_字母表/1024_768/20220401_1024-768zhen+hang+ppt2.lvm")

jmu_frame=readLvm(mix_path)
blue_frame=readLvm(blue_path)
line_frame = readLvm(hang_path)

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

# def replacePoints(data,position,point_num,aim_value):
#     '''该函数用来在固定的位置处替换point_num个数据
#
#     :param data: 需要处理的数据
#     :param position: 需要替换的位置
#     :param point_num: 替换的点数
#     :param aim_value:替换的目标值
#     :return: 替换后的数据
#     '''
#
#     replace_data = np.copy(data)
#     for i in position:
#         for j in range(point_num):
#             index=i+j+1
#             if( (index>=len(replace_data)) or  (replace_data[index]==aim_value)):
#                 break
#             replace_data[index]=aim_value
#
#     return replace_data

def replacePoints(data,position,point_num,aim_value,deviation=5):
    '''该函数用来替换点

    :param data: 需要处理的数据
    :param position: 需要替换的位置
    :param point_num: 替换的点数
    :param aim_value:替换的目标值
    :return: 替换后的数据
    '''

    replace_data = np.copy(data)
    for i in position:
        for j in range(point_num):
            index=i-deviation + j + 1 #为了避免位置误差向左偏移deviation个点
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


def getLineLength(line_data):
    '''该函数用来获取平均的行长度

    :param line_data:行同步信号
    :return:平均行长度
    '''

    #获取行开始和介绍位置
    positions = getSignalInfornamtion(line_data)

    #行数
    line_num = len(positions)
    #平均行长度
    #-----方法一：不太可行，因为行同步信号也占有一定的长度，这部分也要平均进去------------------
    # sum=0
    # for pos in positions:
    #     sum= sum + pos["len"]
    # avg_len = round(sum / len(positions))
    #--------------方法二：进行整体平均----------------------------------------
    avg_len = round(len(line_data) / len(positions))

    #line_info = {'position':positions,'line_num':line_num,'avg_len':avg_len}

    return avg_len



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


def sliceData(data, ways, fileDir, startID, slice_size,backupDir):
    '''该函数用来划分数据集

    :param data: 需要划分的数据
    :param ways: 存储方式
    :param fileDir: 存放的路径
    :param startID: 换分文件的开始id
    :param slice_size: 划分尺寸
    :param backupDir: 该路径用来保存全零的数据集文件
    :return: 返回最后文件的下一个id
    '''

    begin_position = 0
    end_position = begin_position + slice_size
    result = False
    fileID = startID
    while (result == False):
        if (end_position >= len(data)):
            end_position = len(data)
            result = True
            #break

        split_data = data[begin_position:end_position]
        temps=len(split_data)

        if(judgeNonZero(split_data) and (len(split_data)>=slice_size )):
            #如果不是全零并且达到分割的长度拿去训练
            save_dir = fileDir

        else:
            #如果全零或者长度不够分割长度，则备份，在复现的时候添加到数据集中
            save_dir = backupDir

        file_name = str(fileID)
        writeData(split_data, ways, save_dir, file_name)

        fileID = fileID + 1
        begin_position = end_position
        end_position = begin_position + slice_size

    return fileID

def sliceData1(data, ways,file_name,file_end,  slice_size):
    '''该函数用来划分数据集

    :param data: 需要划分的数据
    :param ways: 存储方式
    :param fileDir: 存放的路径
    :param startID: 换分文件的开始id
    :param slice_size: 划分尺寸
    :param backupDir: 该路径用来保存全零的数据集文件
    :return: 返回最后文件的下一个id
    '''
    startID=0
    begin_position = 0
    end_position = begin_position + slice_size
    result = False
    while (result == False):
        if (end_position >= len(data)):
            end_position = len(data)
            result = True
            #break
        split_data = data[begin_position:end_position]
        temps=len(split_data)
        save_dir = save_file

        filename =file_name+str(startID)+file_end
        writeData(split_data, ways, save_dir, filename)

        startID += 1
        begin_position = end_position
        end_position = begin_position + slice_size


def sliceData_CYtasnet(save_file_CY,data, ways,file_name,file_end,  slice_size):
    '''该函数用来划分数据集

    :param data: 需要划分的数据
    :param ways: 存储方式
    :param fileDir: 存放的路径
    :param startID: 换分文件的开始id
    :param slice_size: 划分尺寸
    :param backupDir: 该路径用来保存全零的数据集文件
    :return: 返回最后文件的下一个id
    '''
    startID=0
    begin_position = 0
    end_position = begin_position + slice_size
    result = False
    while (result == False):
        if (end_position >= len(data)):
            end_position = len(data)
            result = True
            #break
        split_data = data[begin_position:end_position]

        # if (judgeNonZero(split_data)):
        #     temps = len(split_data)
        #     save_dir = save_file_CY
        #
        #     filename = file_name + str(startID)
        #     writeData(split_data, ways, save_dir, filename)
        #     # file_name = fileName + '_' + str(begin_position) + '_' + str(end_position)
        #     # writeData(split_data, ways, fileDir, file_name)

        temps=len(split_data)
        save_dir =save_file_CY

        filename =file_name+str(startID)
        writeData(split_data, ways, save_dir, filename)

        startID += 1
        begin_position = end_position
        end_position = begin_position + slice_size
def judgeNonZero(*datas):
    '''该函数用来判断一个矩阵是否为零矩阵

    :param data:需要判断的矩阵
    :return:返回判断结果
    '''

    result = True
    for data in datas:
        result = result * np.any(data)

    return result

def moveDataSetDiffFile(dataSet_dir, backupPath):
    '''该函数用来移动备份数据集中训练集、测试集、验证集中不同的文件

    :param dataSet_dir: 数据集的路径
    :param backupPath: 备份的路径
    :return:
    '''
    addDir_list = ["tr/", "cv/", "tt/"] #数据集构成：训练集、验证集、测试集
    for addDir in addDir_list:
        #每个路径的拼接
        filePath1 = dataSet_dir + addDir + "s1/"
        filePath2 = dataSet_dir + addDir + "s2/"
        filePath3 = dataSet_dir + addDir + "mix/"
        #转移数据集中所存在的不同文件，并对转移的文件进行备份
        result = moveDiffFile(filePath1, filePath2, filePath3, backupPath)
        print(result)

def moveDiffFile(filePath1,filePath2,filePath3,backupPath):
    '''该函数用来将三个文件夹内的不同文件移动到backupPath所在路径下对应的tr、cv、tt文件夹里

    :param filePath1:文件路径1
    :param filePath2:文件路径2
    :param filePath3:文件路径3
    :param backupPath:文件移动的目标路径
    :return: 返回删除的文件名
    '''

    #获取文件内的文件名
    list1 = os.listdir(filePath1)
    list2 = os.listdir(filePath2)
    list3 = os.listdir(filePath3)

    #找到两个文件夹相同的文件
    two_folders_same = [x for x in list2 if x in list1]
    #找到三个文件件的相同文件
    three_folders_same = [x for x in list3 if x in two_folders_same]
    #根据相同的文件，推出每个文件夹中不同的文件
    folder1_diff = [y for y in list1 if y not in three_folders_same]
    folder2_diff = [y for y in list2 if y not in three_folders_same]
    folder3_diff = [y for y in list3 if y not in three_folders_same]
    # 然后将不同文件移动到备份文件夹中
    #backupPath = "../voice_data/backup/"
    #提取tt or cv or tr / s1 or s2  or mix
    subDir_list1 = (filePath1.split('/'))[-3:-1]
    subDir_list2 = (filePath2.split('/'))[-3:-1]
    subDir_list3 = (filePath3.split('/'))[-3:-1]
    #拼接获取移动的目标路径
    dstpath1 = stitchPath(backupPath,subDir_list1)
    dstpath2 = stitchPath(backupPath,subDir_list2)
    dstpath3 = stitchPath(backupPath,subDir_list3)
    #如果路径不存在就新建路径 （这里替换成直接在移动文件的时候判断）
    #makeDirs(dstpath1,dstpath2,dstpath3)
    #实现移动备份
    for i in folder1_diff:
        #os.remove(os.path.join(filePath1, i))
        srcfile = os.path.join(filePath1, i)
        movefile(srcfile, dstpath1)
    for i in folder2_diff:
        #os.remove(os.path.join(filePath2, i))
        srcfile = os.path.join(filePath2, i)
        movefile(srcfile, dstpath2)
    for i in folder3_diff:
        #os.remove(os.path.join(filePath3, i))
        srcfile = os.path.join(filePath3, i)
        movefile(srcfile, dstpath3)

    move_file={"folder1":folder1_diff,"folder2":folder2_diff,"folder3":folder3_diff}
    return move_file

def stitchPath(bashPath,stitc_seg_list):
    '''该函数用来组装一个路径

    :param bashPath: 基础的路径
    :param stitc_seg_list: 需要进行组装的路劲段，以列表的形式给出
    :return:
    '''

    stitch_result = bashPath
    for seg in stitc_seg_list :
        stitch_result += seg+'/'

    return stitch_result

def makeDirs(*dirs):
    '''该函数用来当路径不存在时就新建路径文件夹

    :param dirs: 需要判断的路径
    :return:
    '''
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)

def movefile(srcfile, dstpath):  # 移动函数
    '''该函数用来实现文件夹的移动

    :param srcfile: 需要复制、移动的文件路径
    :param dstpath: 目的地址
    :return:
    '''
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)  # 创建路径
        shutil.move(srcfile, dstpath + fname)  # 移动文件
        print("move %s -> %s" % (srcfile, dstpath + fname))

def dealWavToTxt(wav_path,txt_path):
    '''该函数用来将wav文件转化成txt文件（会有精度损失）

    :param wav_path:需要转换的wav文件
    :param txt_path:保存txt的路径
    :return:
    '''
    data = soundfile.read(wav_path)[0]
    np.savetxt(txt_path, data, fmt='%.05f')

def copyDir(original, target):
    '''将一个目录下的全部文件和目录,完整地<拷贝并覆盖>到另一个目录

    :param original: 源目录
    :param target: 目标目录
    :return:
    '''

    if not os.path.exists(original):
        # 如果传进来的不是目录
        print("传入的原文件夹路径不存在")
        return

    if not os.path.exists(target):
        os.makedirs(target)  # 创建路径

    for a in os.walk(original):
        # 递归创建目录
        for d in a[1]:
            dir_path = os.path.join(a[0].replace(original, target), d)
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)
        # 递归拷贝文件
        for f in a[2]:
            dep_path = os.path.join(a[0], f)
            arr_path = os.path.join(a[0].replace(original, target), f)
            shutil.copy(dep_path, arr_path)



def fen_hang1(file_read_name,file_end,zhen_cs,feng_hang):#几帧，几份 按照份数分
    for cs in range(zhen_cs):
        print("开始feng_hang", cs)
        file = open("./data/"+(fbl)+"/"+str(cs)+"帧"+file_read_name, "r")  # 以只读模式读取文件
        lines = []
        i = 0
        for i in file:
            lines.append(i)  # 逐行将文本存入列表lines中
        file.close()
        new = []
        hang = 0
        for line in lines:  # 逐行遍历
            new.append(re.findall(r'\-?\d+\.?\d*', line))
            hang += 1
        # 以写的方式打开文件，文件不存在，自动创建，如果存在就会覆盖原文件
        # if (hang % feng_hang != 0):
        #     hang += (hang//feng_hang-hang%feng_hang)
        ix = 0
        for ix in range(int(feng_hang)):
            str_ix = str(ix)
            if not os.path.exists(save_file):
                os.makedirs(save_file)
            file_write_obj = open(save_file+"/"+(fbl)+"_"+str(cs)+"帧" + str_ix + file_end, 'w')
            for var in new[ix * hang // feng_hang:(ix + 1) * hang // feng_hang]:
                file_write_obj.writelines(var)
                file_write_obj.writelines('\n')
            file_write_obj.close()


def fen_hang2(file_read_name,file_end,zhen_cs,feng_hang):#几帧，几份 按照每份大小分
    for cs in range(zhen_cs):
        print("开始feng_hang", cs)
        file = open("./data/"+(fbl)+"/"+str(cs)+"帧"+file_read_name, "r")  # 以只读模式读取文件
        lines = []
        i = 0
        for i in file:
            lines.append(i)  # 逐行将文本存入列表lines中
        file.close()
        new = []
        hang = 0
        for line in lines:  # 逐行遍历
            new.append(re.findall(r'\-?\d+\.?\d*', line))
            hang += 1
        # 以写的方式打开文件，文件不存在，自动创建，如果存在就会覆盖原文件
        # if (hang % feng_hang != 0):
        #     hang += (hang//feng_hang-hang%feng_hang)
        ix = 0
        for ix in range(hang//int(feng_hang)):
            str_ix = str(ix)
            if not os.path.exists(save_file):
                os.makedirs(save_file)
            file_write_obj = open(save_file+"/"+(fbl)+"_"+str(cs)+"帧" + str_ix + file_end, 'w')
            for var in new[ix *feng_hang:(ix + 1) *feng_hang]:
                file_write_obj.writelines(var)
                file_write_obj.writelines('\n')
            file_write_obj.close()


def suppleReappearDataSet():
    '''该函数用来将复现数据集补充完整（补上之前去掉的全零数据）

    :return:
    '''
    # 分离前的复现数据集
    target = "./data/signal/original"
    original = "../exp/tmp/out/signal/original"
    copyDir(original, target)
    original = "../voice_data/txt_backup/tt"
    copyDir(original, target)
    # 分离后的复现数据集
    target = "./data/signal/estimate"
    original = "../exp/tmp/out/signal/estimate"
    copyDir(original, target)
    target = "./data/signal/estimate/s1"
    original = "../voice_data/txt_backup/tt/s1"
    copyDir(original, target)
    target = "./data/signal/estimate/s2"
    original = "../voice_data/txt_backup/tt/s2"
    copyDir(original, target)

def getReappearFile(signal_dir,save_dir,save_file_name):
    '''该函数用来获取复现文件

    :param signal_dir: 原数据集路径
    :param save_dir: 复现文件的保存路径
    :param save_file_name:复现文件名字
    :return:
    '''

    #将数据集合成组合成能够复现的文件
    # estimate    original
    # signal_dir = "./data/signal/estimate/s1/"
    # estimate_s1_save_dir ="../voice_data/wav/tt/s1/"
    file_name = os.listdir(signal_dir)
    sort_file_name = sorted(file_name, key=lambda i: int(i.split(".")[0]))

    temp_data = []
    for name in sort_file_name:
        temp = np.loadtxt(signal_dir + name)
        temp_data = np.append(temp_data, temp)
        # temp_data.shape(temp_data_len,1)
        # print(temp_data)

    data = temp_data.reshape(len(temp_data), 1)
    ways = "lvm"
    # save_dir = "./data/"
    # save_file_name = "estimate_s1"
    writeData(data, ways, save_dir, save_file_name)

def getDataset1():
    '''该函数用来获取数据集
    :return:
    '''
    bin_deviation = 0  # 进行二值化的误差
    # 获取信号的帧位置信息
    jmu_info = getSignalInfornamtion(jmu_frame['frame'], bin_deviation)
    # black_info = getSignalInfornamtion(black_frame['frame'], bin_deviation)
    blue_info = getSignalInfornamtion(blue_frame['frame'], bin_deviation)
    line_info = getSignalInfornamtion(line_frame['frame'], bin_deviation)
    # part_num = np.min([len(jmu_info), len(black_info), len(blue_info)])
    part_num = np.min([len(jmu_info), len(blue_info)])
    for zhen_cs in range(part_num):
        print(zhen_cs,"开始")
        # fn = 0  # 信号的帧号
        # 获取统一的长度
        fn=zhen_cs

        if (fn < 2):
            subDir2 = "tt/"
        elif (fn < 8):
            subDir2 = "tr/"
        else:
            subDir2 = "cv/"

        subDir3_1 = "s1/"
        subDir3_2 = "s2/"
        subDir3_3 = "mix/"

        subDir_wav="../voice_data/wav/"
        subDir_txt="../voice_data/txt/"

        filePath_wav_signal = subDir_wav + subDir2 + subDir3_1
        filePath_wav_noise = subDir_wav + subDir2 + subDir3_2
        filePath_wav_mix = subDir_wav + subDir2 + subDir3_3

        filePath_txt_signal = subDir_txt + subDir2 + subDir3_1
        filePath_txt_noise = subDir_txt + subDir2 + subDir3_2
        filePath_txt_mix = subDir_txt + subDir2 + subDir3_3

        # max_len = findMaxLength(jmu_info[fn]['len'], black_info[fn]['len'], blue_info[fn]['len'])
        max_len = findMaxLength(jmu_info[fn]['len'], blue_info[fn]['len'])
        # 获取第一帧的信号
        jmu = jmu_frame['signal'][jmu_info[fn]['start']:jmu_info[fn]['start'] + max_len]
        # black = black_frame['signal'][black_info[fn]['start']:black_info[fn]['start'] + max_len]
        blue = blue_frame['signal'][blue_info[fn]['start']:blue_info[fn]['start'] + max_len]
        line = line_frame['signal'][line_info[fn]['start']:line_info[fn]['start'] + max_len]


        bin_standard = "average"  # 进行二值化的标准
        replace_point = 1  # 进行替换时定位点的值
        point_num = 10  # 替换点的数量
        aim_value = 1  # 替换值
        # 获取相应的掩码
        mask_blue = getMarsk(blue, bin_standard, bin_deviation, replace_point, point_num, aim_value)
        # 利用叠加的掩码文件从jmu信号中获取所需的理想纯净信号

        mask_hang=getMarsk(line, bin_standard, bin_deviation, replace_point, point_num, aim_value)
        # 从jmu中获取理想的噪声信号
        reverse_mask_blue = reverseMatrix(mask_blue)
        blue_noise_blank = reverse_mask_blue * jmu
        distance = 200
        blue_noise = fillBlanks(mask_blue, blue_noise_blank, distance)
        # 数据集的制作
        hang = mask_hang * jmu
        blue_pure = mask_blue * jmu
        s1 = blue_pure
        c12 = blue_pure+hang
        s2 = blue_noise
        mix = jmu
        other=jmu-hang-blue_pure
        line_avg_len = getLineLength(line)
        print("每行", line_avg_len)
        slice_size = line_avg_len * hang_cs
        # np.savetxt("./data/" + str(zhen_cs) + "帧" + "b1_fill_blanks" + ".txt", jmu, fmt='%.05f')

        if subDir2 == "tt/":
            writeData(jmu, "wav", filePath_wav_mix, str(zhen_cs) + "帧")
            writeData(c12, "wav", filePath_wav_signal, str(zhen_cs) + "帧")
            writeData(blue_noise, "wav", filePath_wav_noise, str(zhen_cs) + "帧")
            #
            # writeData(jmu, "txt", filePath_txt_mix, str(zhen_cs) + "帧" )
            # writeData(c12, "txt", filePath_txt_signal, str(zhen_cs) + "帧")
            # writeData(blue_noise, "txt", filePath_txt_noise, str(zhen_cs) + "帧")
            # sliceData_CYtasnet(filePath_wav_mix, mix, "wav", file_name=fbl + str(zhen_cs) + "帧", file_end=".stem",
            #                    slice_size=slice_size)
            # sliceData_CYtasnet(filePath_wav_signal, s1, "wav", file_name=fbl + str(zhen_cs) + "帧",
            #                    file_end=".stem_signal", slice_size=slice_size)
            # sliceData_CYtasnet(filePath_wav_noise, s2, "wav", file_name=fbl + str(zhen_cs) + "帧",
            #                    file_end=".stem_noise", slice_size=slice_size)

            # sliceData_CYtasnet(filePath_txt_mix, mix, "txt", file_name=fbl + str(zhen_cs) + "帧", file_end=".stem",
            #                    slice_size=slice_size)
            # sliceData_CYtasnet(filePath_txt_signal, s1, "txt", file_name=fbl + str(zhen_cs) + "帧",
            #                    file_end=".stem_signal", slice_size=slice_size)
            # sliceData_CYtasnet(filePath_txt_noise, s2, "txt", file_name=fbl + str(zhen_cs) + "帧",
            #                    file_end=".stem_noise", slice_size=slice_size)
        else:
            sliceData_CYtasnet(filePath_wav_mix, mix, "wav", file_name=fbl + str(zhen_cs) + "帧", file_end=".stem",
                               slice_size=slice_size)
            sliceData_CYtasnet(filePath_wav_signal, s1, "wav", file_name=fbl + str(zhen_cs) + "帧",
                               file_end=".stem_signal", slice_size=slice_size)
            sliceData_CYtasnet(filePath_wav_noise, s2, "wav", file_name=fbl + str(zhen_cs) + "帧",
                               file_end=".stem_noise", slice_size=slice_size)


        # writeData(jmu, "wav", filePath_wav_mix, str(zhen_cs) + "帧")
        # writeData(c12, "wav", filePath_wav_signal, str(zhen_cs) + "帧")
        # writeData(blue_noise, "wav", filePath_wav_noise, str(zhen_cs) + "帧")

        # writeData(jmu, "txt", filePath_txt_mix, str(zhen_cs) + "帧" )
        # writeData(c12, "txt", filePath_txt_signal, str(zhen_cs) + "帧")
        # writeData(blue_noise, "txt", filePath_txt_noise, str(zhen_cs) + "帧")
            # sliceData_CYtasnet(filePath_txt_mix,mix, "txt", file_name=fbl + str(zhen_cs) + "帧", file_end=".stem", slice_size=slice_size)
            # sliceData_CYtasnet(filePath_txt_signal,s1, "txt", file_name=fbl+str(zhen_cs) + "帧", file_end=".stem_signal", slice_size=slice_size)
            # sliceData_CYtasnet(filePath_txt_noise,s2, "txt", file_name=fbl+str(zhen_cs) + "帧", file_end=".stem_noise", slice_size=slice_size)


        # writeData(jmu, "txt",  zhen_savepath, str(zhen_cs) + "帧" + "cf_fina1")
        # writeData(c12, "txt", zhen_savepath, str(zhen_cs) + "帧" + "cf_finc12")
        # # writeData(hang, "txt", zhen_savepath, str(zhen_cs) + "帧" + "cf_finhang")
        # # writeData(blue_pure, "txt", zhen_savepath, str(zhen_cs) + "帧" + "cf_finc1")
        # writeData(blue_noise, "txt", zhen_savepath, str(zhen_cs) + "帧" + "b1_fill_blanks")
        # # writeData(other, "txt", zhen_savepath, str(zhen_cs) + "帧" + "other")
        # # 生成数据集路径

        # fen_hang2(file_read_name="cf_fina1.txt",file_end=".stem.txt",zhen_cs=fn,feng_hang=slice_size)
        # fen_hang2(file_read_name="cf_finc1.txt", file_end=".stem_signal.txt", zhen_cs=fn, feng_hang=slice_size)
        # fen_hang2(file_read_name="b1_fill_blanks.txt", file_end=".stem_noise.txt", zhen_cs=fn, feng_hang=slice_size)
        #
        # sliceData1(mix, "txt",file_name=fbl+str(zhen_cs) + "帧",file_end=".stem",  slice_size=slice_size)
        # sliceData1(s1, "txt", file_name=fbl+str(zhen_cs) + "帧", file_end=".stem_signal", slice_size=slice_size)
        # sliceData1(s2, "txt", file_name=fbl+str(zhen_cs) + "帧", file_end=".stem_noise", slice_size=slice_size)
        # sliceData1(hang, "txt", file_name=fbl + str(zhen_cs) + "帧", file_end=".stem_hang", slice_size=slice_size)
        # sliceData1(c12, "txt", file_name=fbl + str(zhen_cs) + "帧", file_end=".stem_c12", slice_size=slice_size)
        # sliceData1(other, "txt", file_name=fbl + str(zhen_cs) + "帧", file_end=".stem_other", slice_size=slice_size)

    print("结束")
#--------------------------------------------------------------------------------------------------------------------------------
#获取数据集
# getDataset1()

# 先将数据集合并完整
# suppleReappearDataSet()
# # 获取复现文件
# #  "./data/signal/estimate/s1/"    "estimate_s1"
# #  "./data/signal/estimate/s2/"   "estimate_s2"
# # "./data/signal/original/s1/"    "original_s1"
# # "./data/signal/original/s2/"    "original_s2"
# signal_dir = "./data/signal/estimate/s1/"
# save_dir = "./data/reappear/"
# save_file_name = "estimate_s1"
# getReappearFile(signal_dir,save_dir,save_file_name)




aaa=np.loadtxt("联想a机外网_0.050帧cf_fina1.txt")

# writeData(aaa, "wav"," ./data/",  "50hz")
soundfile.write("联想a机外网_0.050帧cf_fina1.txt.wav", aaa[0:4135481],44100 , format='wav', subtype='PCM_16')