import matplotlib.pyplot as plt

x_data = ['tanh', 'leaky_relu', 'linear', 'elu', 'swish', 'softmax', 'sigmoid', 'relu']

si_sdr_value = [4.154759242497861, 5.294657337379431, 5.170298644054426, 5.314645773497799, 5.378221149556339, 4.045547847996415, 4.9529901392142435, 5.346545478546154]
sir_value = [20.773744548305793, 28.45552058829892, 28.41304039163173, 28.484107688920357, 28.69726480411575, 25.400299317682187, 27.46134167720215, 29.115855057042552]
sar_value = [6.850831839588111, 7.793211839078924, 7.599877196887756, 7.858430238410758, 7.874855585121274, 6.7827822641668805, 7.547328508022822, 7.777119570608587]
sdr_value = [6.006882978060674, 7.132118306929867, 6.91112708618345, 7.168926753668634, 7.206901241826945, 5.9296528485298206, 6.798750621473524, 7.1219209168065065]

plt.plot(x_data, si_sdr_value, marker='o', label='SI-SDR')
plt.plot(x_data, sir_value, marker='v', label='SIR')
plt.plot(x_data, sar_value, marker='^', label='SAR')
plt.plot(x_data, sdr_value, marker='s', label='SDR')

plt.legend()
plt.xlabel('Activation Function')
plt.ylabel('Value')
plt.title('Performance Comparison of Activation Functions')

plt.xticks(size=12)
plt.yticks(size=12)
plt.show()