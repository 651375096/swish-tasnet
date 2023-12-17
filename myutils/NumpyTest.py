import matplotlib.pyplot as plt
i=0
#x=[1,2,3,4,5]
y1=[1,2,3,4,5]
y2=[1,4,9,16,25] # 要绘制的是（x,y1）和（x,y2）
# subplot(在窗口中分的行、列、画图序列数)
plt.subplot(2,1,1) #第1个图画在一个两行一列分割图的第1幅位置
plt.plot(y1)
plt.subplot(2,1,2) #第2个图画在一个两行一列分割图的第2幅位置
plt.plot(y2)
plt.show()



