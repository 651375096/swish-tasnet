import matplotlib.pyplot as plt

x_data = ['TCN block0', 'TCN block1', 'TCN block2', 'TCN block3', 'TCN block4']


# TCN block0\ &7.1682  & 33.2676 & 9.4126   &8.9228   \\
# TCN block1\ & 7.0671 & 33.5606 & 9.2957   & 8.8222  \\
# TCN block2\ & 7.1230 & 33.2998 & 9.3829   & 8.9068  \\
# TCN block3\ & 7.2199 & 33.5235 & 9.4550   & 9.0013  \\
# TCN block4\ & 7.0235  & 32.8731 & 9.3252   & 8.8274

si_sdr_value = [7.1682,7.0671 ,7.1230,7.2199,7.0235]
sir_value = [33.2676,33.5606,33.2998,33.5235,32.8731]
sar_value = [9.4126,9.2957 ,9.3829,9.4550 ,9.3252  ]
sdr_value = [8.9228 ,8.8222,8.9068,9.0013 ,8.8274  ]

plt.plot(x_data, si_sdr_value, marker='o', label='SI-SDR')
# plt.plot(x_data, sir_value, marker='v', label='SIR')
plt.plot(x_data, sar_value, marker='^', label='SAR')
plt.plot(x_data, sdr_value, marker='s', label='SDR')

plt.legend()
# plt.xlabel('Activation Function')
# plt.ylabel('Value')
plt.title('Separation Performance of Different TCN Blocks')

plt.xticks(size=12)
plt.yticks(size=12)
plt.show()