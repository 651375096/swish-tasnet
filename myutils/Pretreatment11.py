import lvm_read as lvm
import numpy as np
import matplotlib.pyplot as plt
import soundfile


def write_wav(data, path):  # 将信号保存到指定路径
    # （wav格式，PCM编码，单通道，采样率16KHz，位深16bit）？？？
   soundfile.write(path, data, 44100, format='wav', subtype='PCM_16')
# e = [31016,4166498,4197514,8332995,8364011,12499492,12530508,16665990,16697006,20832487,20863504,24998985,25030001,29165482,29196499,33331980,33362996,37498477,37529493,41664975,41695991,45831472]
# f = [31016,4166497,4197513,8332994,8364011,12499491,12530508,16665988,16697004,20832485,20863501,24998982,25029999,29165479,29196495,33331976,33362992,37498473,37529489,41664970,41695986,45831467]
# g = [31016,4166498,4197514,8332995,8364012,12499493,12530509,16665991,16697007,20832488,20863505,24998987,25030003,29165485,29196501,33331982,33362999,37498480,37529497,41664978,41695994,45831476]
#学长的22
# a = lvm.read("F:\\20210605\\1024_768_60_JMU_250MHz.lvm")
# b = lvm.read("F:\\20210605\\1024_768_60_black_250MHz.lvm")
# c = lvm.read("F:\\20210605\\1024_768_60_B_250MHz.lvm")
# d = lvm.read("F:\\20210605\\1024_768_60_ZHENHANG_250MHz.lvm")
# a = lvm.read("E:\\capstone\\jmuData\\20210621\\1024_768_集美大学\\1024_768_60_JMU_250MHz.lvm")
# b = lvm.read("E:\\capstone\\jmuData\\20210621\\1024_768_集美大学\\1024_768_60_black_250MHz.lvm")#black
# c = lvm.read("E:\\capstone\\jmuData\\20210621\\1024_768_集美大学\\1024_768_60_B_250MHz.lvm")
# d = lvm.read("E:\\capstone\\jmuData\\20210621\\1024_768_集美大学\\1024_768_60_ZHENHANG_250MHz.lvm")

#a = lvm.read("E:\\Files\\code\\pretreatment\\1024_768_JMU\\1024_768_60_JMU_250MHz.lvm")
#b = lvm.read("E:\\Files\\code\\pretreatment\\1024_768_JMU\\1024_768_60_black_250MHz.lvm")
#c = lvm.read("E:\\Files\\code\\pretreatment\\1024_768_JMU\\1024_768_60_B_250MHz.lvm")
#d = lvm.read("E:\\Files\\code\\pretreatment\\1024_768_JMU\\1024_768_60_ZHENHANG_250MHz.lvm")


a = lvm.read("/home/myfl/luotingyang/conv_tasnet/electromagnetic/1024_768_JMU/1024_768_60_JMU_250MHz.lvm")
b = lvm.read("/home/myfl/luotingyang/conv_tasnet/electromagnetic/1024_768_JMU/1024_768_60_black_250MHz.lvm")
c = lvm.read("/home/myfl/luotingyang/conv_tasnet/electromagnetic/1024_768_JMU/1024_768_60_B_250MHz.lvm")
d = lvm.read("/home/myfl/luotingyang/conv_tasnet/electromagnetic/1024_768_JMU/1024_768_60_ZHENHANG_250MHz.lvm")



# a = lvm.read("E:\\capstone\\jmuData\\20210621\\1024_25000000\\1024_768_60_JMU_250MHz.lvm")
# b = lvm.read("E:\\capstone\\jmuData\\20210621\\1024_25000000\\1024_768_60_black_250MHz.lvm")#black
# c = lvm.read("E:\\capstone\\jmuData\\20210621\\1024_25000000\\1024_768_60_B_250MHz.lvm")
# d = lvm.read("E:\\capstone\\jmuData\\20210621\\1024_25000000\\1024_768_60_ZHENHANG_250MHz.lvm")

# a = lvm.read("E:\\capstone\\jmuData\\20210621\\1024_768_集美大学\\1024_768_60_ZHENHANG_250MHz-0.lvm")

# a = lvm.read("E:\\capstone\\jmuData\\20210621\\640_480_集美大学\\640_480_60_集美大学_250MHz.lvm")
# b = lvm.read("E:\\capstone\\jmuData\\20210621\\640_480_集美大学\\640_480_60_black_250MHz.lvm")
# c = lvm.read("E:\\capstone\\jmuData\\20210621\\640_480_集美大学\\640_480_60_B_250MHz.lvm")
# d = lvm.read("E:\\capstone\\jmuData\\20210621\\640_480_集美大学\\640_480_60_ZHENHANG_250MHz.lvm")

# a = lvm.read("E:\\capstone\\jmuData\\20210621\\1024\\1024_768_60_JMU_250MHz.lvm")
# b = lvm.read("E:\\capstone\\jmuData\\20210621\\1024\\1024_768_60_black_250MHz.lvm")
# c = lvm.read("E:\\capstone\\jmuData\\20210621\\1024\\1024_768_60_B_250MHz.lvm")
# d = lvm.read("E:\\capstone\\jmuData\\20210621\\1024\\new_1024_768_60_ZHENHANG_250MHz.lvm")

# a = lvm.read("E:\\capstone\\jmuData\\20210621\\640_10000000\\640_480_60_JMU_250MHz.lvm")
# b = lvm.read("E:\\capstone\\jmuData\\20210621\\640_10000000\\640_480_60_black_250MHz.lvm")
# c = lvm.read("E:\\capstone\\jmuData\\20210621\\640_10000000\\640_480_60_B_250MHz.lvm")
# d = lvm.read("E:\\capstone\\jmuData\\20210621\\640_10000000\\640_480_60_ZHENHANG_250MHz.lvm")


#帧预处理
#----------------------a帧预处理-------------------
a_z = a[0]
a_z = a_z['data']
a_z = np.array(a_z)   #生成序列
# a = a[:,0:1]    #JMU第一路混合信号
a_z = a_z[:,1:2]
a_max = max(a_z)
a_min = min(a_z)
a_m = (a_max+a_min)/2
a_z[np.where(a_z<a_m)] = 0  #二值化
a_z[np.where(a_z!=0)] = 1
# plt.plot(a_z)
# plt.title("binaryzation a_z")
# plt.show()

i=0                     #差分
while i < len(a_z) - 1:
    a_z[i] = a_z[i+1] - a_z[i]
    i = i + 1
# plt.plot(a_z)
# plt.title("difference a_z")
# plt.show()
print(min(a_z))
print(max(a_z))

a_index = np.where(a_z == -1)
a_index2 = np.where(a_z == 1)
print(a_index)
print(a_index2)

a_first = int(a_index[0][0])
a_final = int(a_index[0][-1])
print(a_first)
print(a_final)

#---------------------------b帧预处理-------------------
b_z = b[0]
b_z = b_z['data']
b_z = np.array(b_z)   #生成序列
# a = a[:,0:1]    #JMU第一路混合信号
b_z = b_z[:,1:2]
b_max = max(b_z)
b_min = min(b_z)

print("b_max")
print(b_max)
print("b_min")
print(b_min)

b_m = (b_max+b_min)/2
print("b_m")
print(b_m)
b_z[np.where(b_z<b_m)] = 0  #二值化
b_z[np.where(b_z!=0)] = 1
# plt.plot(b_z)
# plt.title("binaryzation b_z")
# plt.show()

i=0                     #差分
while i < len(b_z) - 1:
    b_z[i] = b_z[i+1] - b_z[i]
    i = i + 1
# plt.plot(b_z)
# plt.title("difference b")
# plt.show()
print(min(b_z))
print(max(b_z))

b_index = np.where(b_z == -1)
b_index2 = np.where(b_z == 1)
print(b_index)
print(b_index2)

b_first = int(b_index[0][0])
b_final = int(b_index[0][-1])
print(b_first)
print(b_final)

#----------------------c帧预处理-------------------
c_z = c[0]
c_z = c_z['data']
c_z = np.array(c_z)   #生成序列
# a = a[:,0:1]    #JMU第一路混合信号
c_z = c_z[:,1:2]
c_max = max(c_z)
c_min = min(c_z)
c_m = (c_max+c_min)/2
c_z[np.where(c_z<c_m)] = 0  #二值化
c_z[np.where(c_z!=0)] = 1
# plt.plot(c_z)
# plt.title("binaryzation c_z")
# plt.show()

i=0                     #差分
while i < len(c_z) - 1:
    c_z[i] = c_z[i+1] - c_z[i]
    i = i + 1
# plt.plot(c_z)
# plt.title("difference c_z")
# plt.show()
print(min(c_z))
print(max(c_z))

c_index = np.where(c_z == -1)
c_index2 = np.where(c_z == 1)
print(c_index)
print(c_index2)

c_first = int(c_index[0][0])
c_final = int(c_index[0][-1])
print(c_first)
print(c_final)

#-----------------------d帧预处理------------------------
d_z = d[0]
d_z = d_z['data']
d_z = np.array(d_z)   #生成序列
# a = a[:,0:1]    #JMU第一路混合信号
d_z = d_z[:,1:2]
d_max = max(d_z)
d_min = min(d_z)
d_m = (d_max+d_min)/2
d_z[np.where(d_z<d_m)] = 0  #二值化
d_z[np.where(d_z!=0)] = 1
# plt.plot(d_z)
# plt.title("binaryzation d_z")
# plt.show()

i=0                     #差分
while i < len(d_z) - 1:
    d_z[i] = d_z[i+1] - d_z[i]
    i = i + 1
# plt.plot(d_z)
# plt.title("difference d_z")
# plt.show()
print(min(d_z))
print(max(d_z))

d_index = np.where(d_z == -1)
d_index2 = np.where(d_z == 1)
print(d_index)
print(d_index2)

d_first = int(d_index[0][0])
d_final = int(d_index[0][-1])
print(d_first)
print(d_final)
#------------------------------LL--------------------------
len_a = a_final - a_first
len_b = b_final - b_first
len_c = c_final - c_first
len_d = d_final - d_first
LL = min(len_a,len_b,len_c,len_d) + 1

d5_z = d_z[d_first:d_first+LL]    #python数组最后一个数不包含
d5_index = np.where(d5_z == -1)
d5_index2 = np.where(d5_z == 1)
print(d5_index)
print(d5_index)

# LL = a_final - a_first

#------------------------a混合信号处理-------------------
a = a[0]
a = a['data']
a = np.array(a)   #生成序列
a = a[:,0:1]    #JMU第一路混合信号

a1 = np.copy(a)
a1 = a1[a_first:a_first+LL]
a1_len = len(a1)
print(a1_len)

# plt.plot(a1)
# plt.title("a1")
# plt.show()
a2 = np.copy(a1)   #a2用于后续得到纯净信号
# print(len(a2))
a3 = np.copy(a1)


#--------------------b混合信号处理----------------------
b = b[0]
b = b['data']
b = np.array(b)
b = b[:,0:1]  #Black第一路混合信号
b1 = np.copy(b)

b1 = b1[b_first:b_first+LL]   #取到第一帧同步

# plt.plot(b1)
# plt.title("b1")
# plt.show()
# b1 = b1[818581:46794516]   #取到第一帧同步    45,975,935



#-----------------------c混合信号处理---------------------
c = c[0]
c = c['data']
c = np.array(c)
c = c[:,0:1]   #取B的第一路混合信号
c1 = np.copy(c)

c5 = c1[c_first:c_first+LL]
# print("c5:")
# print(len(c5))
# plt.plot(c5)
# plt.title("c1")
# plt.show()



#---------------------d混合信号处理--------------------
d = d[0]
d = d['data']
d = np.array(d)
d = d[:,0:1]    #取行同步信号
d1 = np.copy(d)

d5 = d1[d_first:d_first+LL]

# d_index = np.where(d_z == -1)
# d_index2 = np.where(d_z == 1)
d6 = d1[d_index2[0][0]:d_index[0][1]]
d6_max = max(d6)
d6_min = min(d6)
d6_maxz = (d6_max+d6_min)/2
print("d6_maxz:")
print(d6_maxz)

d6[np.where(d6<d6_maxz)] = 0   #二值化
d6[np.where(d6!=0)] = 1
d7 = np.copy(d6)
len_d6 = len(d6)
i = 0         #差分操作
while i < len_d6-1:
    d7[i] = d6[i+1] - d6[i]
    i = i+1

d7_index = np.where(d7 == -1)
d7_index2 = np.where(d7 == 1)
print("d7_index:")
print(d7_index)
print("d7_index2")
print(d7_index2)
hang = len(d7_index2[0])
print("hang:")
print(hang)



print("d5")
print(len(d5))
# plt.plot(d5)
# plt.title("d1")
# plt.show()

#My Code
d_i = 0
d_max2 = max(d5)
d_min2 = min(d5)
d_maxz2 = (d_max2+d_min2)/2
print(d_maxz2)
d5[np.where(d5<d_maxz2)] = 0   #二值化
d5[np.where(d5!=0)] = 1

# plt.plot(d5)
# plt.title("binaryzation-d2")
# plt.show()
d2 = np.copy(d5)  #d2为差分矩阵

len_d2 = len(d2)
i = 0         #差分操作
while i < len_d2-1:
    d2[i] = d5[i+1] - d5[i]
    i = i+1
# d2 = abs(d2)   #d2为差值矩阵
# j = 0
# while j < len_d2-1:   #处理50个位置，50是信号波动大小49999999
#    if (d2[j] == -1):
#       for q in range(9):
#          j = j+1
#          # d2[j] = d2[j]+1
#          d2[j] = 1
#    j = j+1
# d2 = abs(d2)   #d2为差值矩阵

# plt.plot(d2)
# plt.title("difference-d2")
# plt.show()

# d2 = d2[d_first:d_first+LL]
# d2 = d2[d_first:d_final]
# len_a3 = len(a3)
# len_d2 = len(d2)
# if(len_d2 < len_a3):
#     LL = len(d2)
# a3 = a3[0:LL]
# print("length_d:")
# print(len_d2)
# print("a3:")
# print(len(a3))
# a3 = a3*d2      #行同步与JMU混合信号进行操作    ????
# a3 = a3*0.15                 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11

# plt.plot(a3)
# plt.title("a3 = a3*d2")
# plt.show()

c_max2 = max(c5)
c_min2 = min(c5)
c_maxz2 = (c_max2+c_min2)/2
c5[np.where(c5 < c_maxz2)] = 0    #对纯净信号进行二值化
c5[np.where(c5 != 0)] = 1

c3 = np.copy(c5)

# plt.plot(c3)
# plt.title("binaryzation-c3")
# plt.show()

len_c3 = len(c3)
i = 0         #差分操作
while i < len_c3-1:
    c3[i] = c5[i+1] - c5[i]
    i = i+1

# plt.plot(c3)
# plt.title("difference-c3")
# plt.show()

print("d2:")
print(len(d2))
print("c3:")
print(len(c3))
mix_cd = d2+c3

# plt.plot(mix_cd)
# plt.title("mix_cd")
# plt.show()

# j = 0
# while j < len_d2-1:   #处理50个位置，50是信号波动大小49999999
#    if (j in range(d_index[0])):
#       for q in range(9):
#          j = j+1
#          # d2[j] = d2[j]+1
#          mix_cd[j] = 1
#    j = j+1
print("d5_index[0]:")
print(d5_index[0])

j=0
for j in d5_index[0]:
    # cd_index = j - d5_index[0][0]
    for q in range(9):
        mix_cd[j+q] = 1

# plt.plot(mix_cd)
# plt.title("mix_cd-10")
# plt.show()

mix_acd = mix_cd*a2

np.savetxt("E:\\Files\\code\\pretreatment\\mix_acd.txt",mix_acd,fmt='%.05f')

# plt.plot(mix_acd)
# plt.title("mix_acd-10")
# plt.show()


len_c3 = len(c3)
# i = 0            #纯净信号差分操作
# while i < len_c3-1:
#     c3[i] = c1[i+1] - c1[i]
#     i = i+1
#
# plt.plot(c3)
# plt.title("difference-c3")
# plt.show()
#
# # c3 = abs(c3)   #C3为差值矩阵
# # j = 0
# # while j < len_c3-1:
# #    if (c3[j] == -1):
# #       for q in range(9):
# #          j = j+1
# #          # c3[j] = c3[j]+1
# #          c3[j] = 1
# #    j = j+1
# # c3 = abs(c3)   #C3为差值矩阵
# c3 = c3[c_first:c_first+LL]
#
# plt.plot(c3)
# plt.title("c3-10")
# plt.show()
# # a2 = a2[0:LL]
# a2 = a2*c3   #operands could not be broadcast together with shapes (4135481,1) (45975935,1)
# # a2 = a2*0.7
#
# plt.plot(a2)
# plt.title("a2 = a2*c3")
# plt.show()
#
# a2_max2 = max(a2)
# a2_min2 = min(a2)
# print("a2_max2:")
# print(a2_max2)
# print("a2_min2:")
# print(a2_min2)
# a2_maxz2 = (a2_max2+a2_min2)/2
# print(a2_maxz2)
# a2[np.where(a2>a2_maxz2)] = a2_maxz2          #???????
# a2[np.where(a2<-a2_maxz2)] = -a2_maxz2
# # a2[np.where(a2>0.002)] = 0.002          #???????
# # a2[np.where(a2<-0.002)] = -0.002
# a2 = a2+a3
#
# # plt.figure(2)
# plt.plot(a2)
# plt.title("Final")
# plt.show()
# # a2 = a2[31016:4166498]
# np.set_printoptions(precision=5)
# # np.savetxt("C:\\Users\\叶\\Desktop\\1024_768_60_B_250MHz_替换2.txt",a2,fmt='%.05f')
#
#
#
#






# a = a[0]
# a = a['data']
# a = np.array(a)   #生成序列
# a = a[:,0:1]    #JMU第一路混合信号
# len_n = len(a)
#
# # li[start : end : step]
# # start是切片起点索引，end是切片终点索引，但切片结果不包括终点索引的值。step是步长默认是1。
#
# a1 = np.copy(a)
#
# #My Code
# a_i = 0
# a_max = max(a1)
# a_maxz = a_max/2
# a_i = np.where(a1 > a_maxz)
# print(a_i)
# # print(a_i[0])
# # print(a_i[-1])
# # print(a_i[0][0])
# # print(a_i[0][-1])
# # # a_i.flatten()   #二维数组转一维
# # print(a_i)
# a1 = a1[a_i[0][0]:a_i[0][-1]]
# # a1 = a1[a_i[0][0]:a_i[0][-1]]
# # print(a_i[0][0])
# # print(a_i[0][-1])
# # print(a1)
# # print(len(a1))
# len_a1 = int(a_i[0][-1] - a_i[0][0])
# # print(len_a1)
# # a1 = a1[1700340:47676275]  #取到第一帧同步   ？？？怎么求45,975,935
#
# a2 = np.copy(a1)   #a2用于后续得到纯净信号
# # print(len(a2))
# a3 = np.copy(a1)
# # a3 = np.copy(a2)   #a3用于后续与行同步做操作
# # a2 = a2[41695991:45831472]        #??????????????
# np.set_printoptions(precision=5)    #调整输出的小数点都保留5位小数
# # np.savetxt("E:\\capstone\\HomeTest\\test1.txt",a2,fmt='%.05f')
# # 保存单个文件为一行/列
# b = b[0]
# b = b['data']
# b = np.array(b)
# b = b[:,0:1]  #Black第一路混合信号
# b1 = np.copy(b)
# #My Code
# b_i = 0
# b_max = max(b1)
# b_maxz = b_max/2
# b_i = np.where(b1 > b_maxz)
# # b_i.flatten()
# b1 = b1[b_i[0][0]:int(b_i[0][0]+len_a1)]   #取到第一帧同步
# len_b1 = int(b_i[0][0]+len_a1)-b_i[0][0]
# print(len_b1)
# # b1 = b1[818581:46794516]   #取到第一帧同步    45,975,935
#
#
# c = c[0]
# c = c['data']
# c = np.array(c)
# c = c[:,0:1]   #取B的第一路混合信号
# c1 = np.copy(c)
#
#
# d = d[0]
# d = d['data']
# d = np.array(d)
# d = d[:,0:1]    #取行同步信号
# d1 = np.copy(d)
#
# #My Code
# d_i = 0
# d_max = max(d1)
# d_maxz = d_max/2
# d1[np.where(d1<d_maxz)] = 0   #二值化
# d1[np.where(d1!=0)] = 1
# # d1[np.where(d1<3)] = 0   #二值化
# # d1[np.where(d1!=0)] = 1
# d2 = np.copy(d1)  #d2为差分矩阵
#
#
# i = 0         #差分操作
# while i < len_n-1:
#     d2[i] = d1[i+1] - d1[i]
#     i = i+1
# d2 = abs(d2)   #d2为差值矩阵
# j = 0
# while j < len_n-1:   #处理50个位置，50是信号波动大小49999999
#    if (d2[j] == 1):
#       for q in range(9):
#          j = j+1
#          d2[j] = d2[j]+1
#    j = j+1
# #My Code
# d_i = 0
# d_max = max(d2)
# d_maxz = d_max/2
# d_i = np.where(d2 > d_maxz)
# d2 = d2[d_i[0][0]:int(d_i[0][0]+len_a1)]
# len_d2 = int(d_i[0][0]+len_a1) - d_i[0][0]
# print(len_d2)
# # d2 = d2[659689:46635624]    #将信号处理成与其他三个信号相同大小    45,975,935
# a3 = a3*d2      #行同步与JMU混合信号进行操作    ????
# a3 = a3*0.15
#
# #My Code
# c_i = 0
# c_max = max(c1)
# c_maxz = c_max/2
# c1[np.where(c1 < c_maxz)] = 0    #对纯净信号进行二值化
# c1[np.where(c1 != 0)] = 1
# # c1[np.where(c1 < 0.8)] = 0    #对纯净信号进行二值化
# # c1[np.where(c1 != 0)] = 1
#
#
# c3 = np.copy(c1)
# i = 0            #纯净信号差分操作
# while i < len_n-1:
#     c3[i] = c1[i+1] - c1[i]
#     i = i+1
# c3 = abs(c3)   #C3为差值矩阵
# j = 0
# while j < len_n-1:
#    if (c3[j] == 1):
#       for q in range(9):
#          j = j+1
#          c3[j] = c3[j]+1
#    j = j+1
# #My Code
# c3_i = 0
# c3_max = max(c3)
# c3_maxz = c3_max/2
# c3_i = np.where(c3 > c3_maxz)
# c3 = c3[c3_i[0][0]:int(c3_i[0][0]+len_a1)]
# len_c3 = int(c3_i[0][0]+len_a1) - c3_i[0][0]
# print(len_c3)
# # c3 = c3[4024065:]
# a2 = a2*c3   #operands could not be broadcast together with shapes (4135481,1) (45975935,1)
#
# a2 = a2*0.7
# #My Code
# a2_i = 0
# a2_max = max(a2)
# a2_maxz = a2_max/2
# print(a2_maxz)
# print(a_maxz)
# a2[np.where(a2>a_maxz)] = a_maxz          #???????
# a2[np.where(a2<-a_maxz)] = -a_maxz
# # a2[np.where(a2>0.002)] = 0.002          #???????
# # a2[np.where(a2<-0.002)] = -0.002
# a2 = a2+a3
#
#
# plt.figure(2)
# plt.plot(a2)
# plt.show()
# # a2 = a2[31016:4166498]
# np.set_printoptions(precision=5)
# # np.savetxt("C:\\Users\\叶\\Desktop\\1024_768_60_B_250MHz_替换2.txt",a2,fmt='%.05f')



# e = [31016,4166498,4197514,8332995,8364011,12499492,12530508,16665990,
# 16697006,20832487,20863504,24998985,25030001,29165482,29196499,33331980,
# 33362996,37498477,37529493,41664975,41695991,45831472]

# d_index2[0][0]:d_index[0][1]
'''
yizhen = d_index[0][1] - d_index2[0][0]
print("yizhen")#4135488
print(yizhen)
yizhen2 = d_index[0][2] - d_index2[0][1]
print("yizhen2")#4135489
print(yizhen2)
hang2 = int(yizhen/len(d7_index2[0]))*70
print("hang2")#5156
print(hang2)

x = 0
while x < len(a_index[0])-1:
    a4 = a1[a_index[0][x]:a_index[0][(x+1)]]
    y = 0
    while y < len(d7_index2[0]):
        a5 = a4[y*hang2:(y+1)*hang2]
        write_wav(a5, 'F:\\学习\\Capstone\\20210621\\deal\\data\\' + str(x) + '_' + str(y) + '.stem.mp4')
        write_wav(a5, 'F:\\学习\\Capstone\\20210621\\deal\\data\\' + str(x) + '_' + str(y) + '.stem_mix.wav')
        y += 1
    x = x+1

x = 0
while x < len(a_index[0])-1:
    b2 = b1[a_index[0][x]:a_index[0][(x+1)]]
    y = 0
    while y < len(d7_index2[0]):
        b3 = b2[y*hang2:(y+1)*hang2]
        write_wav(b3, 'F:\\学习\\Capstone\\20210621\\deal\\data\\' + str(x) + '_' + str(y) + '.stem_accompaniment.wav')
        y += 1
    x = x+1

x = 0
while x < len(a_index[0])-1:
    c4 = a2[a_index[0][x]:a_index[0][(x+1)]]
    y = 0
    while y < len(d7_index2[0]):
        c5 = c4[y*hang2:(y+1)*hang2]
        write_wav(c5, 'F:\\学习\\Capstone\\20210621\\deal\\data\\' + str(x) + '_' + str(y) + '.stem_vocals.wav')
        y += 1
    x = x+1

'''
