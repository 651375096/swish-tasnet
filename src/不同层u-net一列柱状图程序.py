import numpy as np
import matplotlib.pyplot as plt
import random
x_data=[f"{i} " for i in range(1, 9)]
y_data=[0.7437797701533402,0.7922641066223204,0.7792485358825677,0.7795951101690591,0.7999443674995852,0.7791438190259843,0.7654477755508275,0.7757564091631891,0.7604602756730231, 0.7605850955910487]



# 正确显示中文和负号
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
# for i in range(len(x_data)):
#     plt.text(x = i-0.3, y = y_data[i]*1.01,s =round(y_data[i],3))
# # 画图，plt.bar()可以画柱状图

for i in range(len(x_data)):
    plt.bar(x_data[i], y_data[i], width=0.9)
    plt.xticks(fontproperties='Times New Roman', size=12)
    plt.yticks(fontproperties='Times New Roman', size=12)#设置大小及加粗

# 设置图片名称
# plt.title("Blind source separation of electromagnetic signals based on Wave-U-Net")
# 设置x轴标签名
# plt.xlabel("u-net layers")
# 设置y轴标签名
# plt.ylabel("Mean Squared Error",size=12)
# plt.ylabel(" Signal-To-Noise Ratio",size=12)
plt.ylabel("SDR",size=12)
# 显示
plt.show()
# """""#一路柱状图





