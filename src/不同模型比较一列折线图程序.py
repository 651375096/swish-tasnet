import matplotlib.pyplot as plt

x_data = ['Swish-Tasnet', 'Conv-Tasnet', 'Wave-U-Net', 'Full-scale', 'Deep Focusing']

si_sdr_value = [7.2199,7.1682,-17.8668,-15.2794,-13.9289]
sir_value = [33.5235 ,33.2676,-0.1172,-1.1313 ,-0.7107  ]
sar_value = [9.4550,9.4126 ,1.0947 ,-0.4929,0.2439   ]
sdr_value = [9.0013,8.9228,-7.2106 ,-7.6877,-7.3685 ]

plt.plot(x_data, si_sdr_value, marker='o', label='SI-SDR')
# plt.plot(x_data, sir_value, marker='v', label='SIR')
plt.plot(x_data, sar_value, marker='^', label='SAR')
plt.plot(x_data, sdr_value, marker='s', label='SDR')

plt.legend()
# plt.xlabel('Activation Function')
# plt.ylabel('Value')
plt.title('Separation Performance of different Networks ')

plt.xticks(size=12)
plt.yticks(size=12)
plt.show()