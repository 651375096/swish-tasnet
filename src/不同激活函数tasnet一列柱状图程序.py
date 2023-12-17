import matplotlib
import matplotlib.pyplot as plt
import numpy as np

x_data=[ "softmax","tanh","leaky_relu","elu","swish","relu","sigmoid","linear"]

# tanh=[23.80801234407036,5.9638607409023034,5.254666902479855]
# leaky_relu=[29.321820039433103,6.951840374862906,6.310865192274125]
# linear=[29.228841966549847,6.513792575285179,5.860226003167317]
# # elu=[26.309633908785305,7.769748609133537,7.1472304622659015]
# swish=[29.72012676624616,7.145643348839046,6.487511938315304]
# softmax=[ 26.961746391815637,5.913939745397993,5.152922143249935]
# sigmoid=[29.40022669554007,7.005785760691945,6.354073029866168,]
# relu=[30.10052069376614,7.304601181132371,6.670963878017362]


tanh=[4.154759242497861,20.773744548305793,6.850831839588111,6.006882978060674]
leaky_relu=[5.294657337379431,28.45552058829892,7.793211839078924,7.132118306929867]
linear=[5.170298644054426,28.41304039163173,7.599877196887756,6.91112708618345]
elu=[5.314645773497799,28.484107688920357,7.858430238410758,7.168926753668634]
swish=[5.378221149556339,28.69726480411575,7.874855585121274,7.206901241826945]

softmax=[4.045547847996415,25.400299317682187,6.7827822641668805,5.9296528485298206]
sigmoid=[4.9529901392142435,27.46134167720215,7.547328508022822,6.798750621473524]
relu=[5.346545478546154,29.115855057042552, 7.777119570608587,7.1219209168065065,]

si_sdr_value,sir_value,sar_value,sdr_value=[],[],[],[]
for ys in (softmax,tanh,leaky_relu,elu,swish,relu,sigmoid,linear):
    si_sdr_value.append(ys[0])
    sir_value.append(ys[1])
    sar_value.append(ys[2])
    sdr_value.append(ys[3])

print("si_sdr_value", x_data[np.argmax(si_sdr_value)], "层最大", np.max(si_sdr_value))

print("sir_value", x_data[np.argmax(sir_value)], "层最大", np.max(sir_value))
print("sar_value", x_data[np.argmax(sar_value)], "层最大", np.max(sar_value))
print("sdr_value", x_data[np.argmax(sdr_value)], "层最大", np.max(sdr_value))

y_data=(si_sdr_value)
for i in range(len(x_data)):
    plt.bar(x_data[i], y_data[i], width=0.45)
    plt.xticks(size=12)
    plt.yticks(size=12)#设置大小及加粗
plt.ylabel("SI-SDR",size=18)
plt.show()


y_data=(sir_value)
for i in range(len(x_data)):
    plt.bar(x_data[i], y_data[i], width=0.45)
    plt.xticks(size=12)
    plt.yticks(size=12)#设置大小及加粗
plt.ylabel("SIR",size=18)
plt.show()

y_data=(sar_value)
for i in range(len(x_data)):
    plt.bar(x_data[i], y_data[i], width=0.45)
    plt.xticks(size=12)
    plt.yticks(size=12)  # 设置大小及加粗
plt.ylabel("SAR",size=18)
plt.show()

y_data=(sdr_value)
for i in range(len(x_data)):
    plt.bar(x_data[i], y_data[i], width=0.5)
    plt.xticks( size=12)
    plt.yticks( size=12)#设置大小及加粗
plt.ylabel("SDR",size=18)
plt.show()


# 显示
# plt.show()
# """""#一路柱状图

