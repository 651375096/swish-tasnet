    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    x_data=[ "1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16"]
    a1=[0.05766427721328915,15.137638015050992,3.8497795912321706,2.0328967180121595]
    a2=[1.467152133506955,18.75165863171454,4.934290225839894,3.466810125201279]
    a3=[1.9301584835282561,19.561619081427384,5.254755615557878,3.8806294608448137]
    a4=[2.374979731893831,21.557694747276777,5.519669299612595,4.374421754250491]
    a5=[3.287891501317665,24.961297847063804,5.987919841456872,5.086467172439946]
    a6=[3.9781874542782356,25.98853097640014,6.648543012120304,5.86149674700827]
    a7=[4.6053686080395195,27.166149510559823,7.347095696345897,6.5631315513180715]
    a8=[5.378221149556339,28.69726480411575,7.874855585121274,7.206901241826945]
    a9=[5.750243948679608,29.42148728462339,8.21018911785991,7.555031839210308]
    a10=[6.3461922995918725,31.353200717829907,8.754764315788831,8.180097988336255]
    a11=[6.542741033578582,31.648168098618846,8.826308775894733,8.271576661765929]
    a12=[6.649362794828052,32.27603034890824,8.917216900884,8.41334723593292]
    a13=[6.842806699807392,31.836515826393192,9.111581798553775,8.583955666241083]
    a14=[7.168259241804914,33.26761692400623,9.412624715436593,8.922883836150682,]
    a15=[7.103207745505736,33.435708039462774,9.334691371570417,8.831904867285946,]
    # a16=[7.168217212446946,33.404945908191195,9.348082641004877,8.858872395331124,]
    a16=[7.1263,33.404,9.3480,8.8588]




    si_sdr_value,sir_value,sar_value,sdr_value=[],[],[],[]
    for ys in (a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16):
        si_sdr_value.append(ys[0])
        sir_value.append(ys[1])
        sar_value.append(ys[2])
        sdr_value.append(ys[3])

    print("si_sdr_value",x_data[np.argmax(si_sdr_value)],"层最大",np.max(si_sdr_value))

    print("sir_value",x_data[np.argmax(sir_value)],"层最大",np.max(sir_value))
    print("sar_value",x_data[np.argmax(sar_value)],"层最大",np.max(sar_value))
    print("sdr_value",x_data[np.argmax(sdr_value)],"层最大",np.max(sdr_value))

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