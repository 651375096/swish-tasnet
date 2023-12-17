import wave

def MergeWavs(in_path1,in_path2,out_path):

    infiles= [in_path1, in_path2]
    outfile=out_path

    data= []
    for infile in infiles:
        w = wave.open(infile, 'rb')
        data.append( [w.getparams(), w.readframes(w.getnframes())] )
        w.close()

    output = wave.open(outfile, 'wb')
    output.setparams(data[0][0])
    output.writeframes(data[0][1])
    output.writeframes(data[1][1])
    output.close()

#path = input('请输入文件路径(结尾加上/)：')

#path="E:\\Files\\code\\electromagnetism\\tr\\s1\\"

str1=["tr","cv"]
str2=["s1","s2","mix"]

for s1 in str1:
    for s2 in str2:
        path="E:\\Files\\code\\electromagnetism\\"+s1+"\\"+s2+"\\"
        i = 1
        num = 0
        while i < 10:
            num = num + 1
            in_path1 = path + "a" + str(num) + ".wav"
            num = num + 1
            in_path2 = path + "a" + str(num) + ".wav"
            out_path = path + "b" + str(i) + ".wav"
            MergeWavs(in_path1, in_path2, out_path)
            i = i + 1

print("finish\n")


