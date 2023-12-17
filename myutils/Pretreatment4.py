import lvm_read as lvm
import numpy as np
import matplotlib.pyplot as plt
import soundfile


def write_wav(data, path):  # 将信号保存到指定路径

    soundfile.write(path, data, 44100, format='wav', subtype='PCM_16')

e = [31016,4166498,4197514,8332995,8364011,12499492,12530508,16665990,16697006,20832487,20863504,24998985,25030001,29165482,29196499,33331980,33362996,37498477,37529493,41664975,41695991,45831472]
f = [31016,4166497,4197513,8332994,8364011,12499491,12530508,16665988,16697004,20832485,20863501,24998982,25029999,29165479,29196495,33331976,33362992,37498473,37529489,41664970,41695986,45831467]
g = [31016,4166498,4197514,8332995,8364012,12499493,12530509,16665991,16697007,20832488,20863505,24998987,25030003,29165485,29196501,33331982,33362999,37498480,37529497,41664978,41695994,45831476]
a = lvm.read("F:\\20210605\\1024_768_60_JMU_250MHz.lvm")
b = lvm.read("F:\\20210605\\1024_768_60_black_250MHz.lvm")
c = lvm.read("F:\\20210605\\1024_768_60_B_250MHz.lvm")
d = lvm.read("F:\\20210605\\1024_768_60_ZHENHANG_250MHz.lvm")
a = a[0]
a = a['data']
a = np.array(a)
a = a[:,0:1]    #JMU第一路混合信号
a1 = np.copy(a)
a1 = a1[1700340:47676275]  #取到第一帧同步
a2 = np.copy(a1)   #a2用于后续得到纯净信号
a3 = np.copy(a2)   #a3用于后续与行同步做操作
a2 = a2[41695991:45831472]
np.set_printoptions(precision=5)
np.savetxt("C:\\Users\\叶\\Desktop\\123123.txt",a2,fmt='%.05f')

b = b[0]
b = b['data']
b = np.array(b)
b = b[:,0:1]  #Black第一路混合信号
b1 = np.copy(b)
b1 = b1[818581:46794516]   #取到第一帧同步


c = c[0]
c = c['data']
c = np.array(c)
c = c[:,0:1]   #取B的第一路混合信号
c1 = np.copy(c)


d = d[0]
d = d['data']
d = np.array(d)
d = d[:,0:1]    #取行同步信号
d1 = np.copy(d)


d1[np.where(d1<3)] = 0   #二值化
d1[np.where(d1!=0)] = 1
d2 = np.copy(d1)  #d2为差分矩阵
i = 0         #差分操作
while i < 49999999:
    d2[i] = d1[i+1] - d1[i]
    i = i+1
d2 = abs(d2)   #d2为差值矩阵
j = 0
while j < 49999999:   #处理50个位置，50是信号波动大小
   if (d2[j] == 1):
      for q in range(9):
         j = j+1
         d2[j] = d2[j]+1
   j = j+1
d2 = d2[659689:46635624]    #将信号处理成与其他三个信号相同大小
a3 = a3*d2      #行同步与JMU混合信号进行操作
a3 = a3*0.15

c1[np.where(c1 < 0.8)] = 0    #对纯净信号进行二值化
c1[np.where(c1 != 0)] = 1


c3 = np.copy(c1)
i = 0            #纯净信号差分操作
while i < 49999999:
    c3[i] = c1[i+1] - c1[i]
    i = i+1
c3 = abs(c3)   #C3为差值矩阵
j = 0
while j < 49999999:
   if (c3[j] == 1):
      for q in range(9):
         j = j+1
         c3[j] = c3[j]+1
   j = j+1
c3 = c3[4024065:]
a2 = a2*c3

a2 = a2*0.7
a2[np.where(a2>0.002)] = 0.002
a2[np.where(a2<-0.002)] = -0.002
a2 = a2+a3


plt.figure(2)
plt.plot(a2)
plt.show()
a2 = a2[31016:4166498]
np.set_printoptions(precision=5)
np.savetxt("C:\\Users\\叶\\Desktop\\1024_768_60_B_250MHz_替换2.txt",a2,fmt='%.05f')



x = 0
while x < 22:
    a4 = a1[e[x]:e[(x+1)]]
    y = 0
    while y < 11:
        a5 = a4[y*372000:(y+1)*372000]
        write_wav(a5, 'E:\\data\\' + str(x) + '_' + str(y) + '.stem.mp4')
        write_wav(a5, 'E:\\data\\' + str(x) + '_' + str(y) + '.stem_mix.wav')
        y += 1
    x += 2

x = 0
while x < 22:
    b2 = b1[f[x]:f[(x+1)]]
    y = 0
    while y < 11:
        b3 = b2[y*372000:(y+1)*372000]
        write_wav(b3, 'E:\\data\\' + str(x) + '_' + str(y) + '.stem_accompaniment.wav')
        y += 1
    x += 2

x = 0
while x < 22:
    c4 = a2[g[x]:g[(x+1)]]
    y = 0
    while y < 11:
        c5 = c4[y*372000:(y+1)*372000]
        write_wav(c5, 'E:\\data\\' + str(x) + '_' + str(y) + '.stem_vocals.wav')
        y += 1
    x += 2


