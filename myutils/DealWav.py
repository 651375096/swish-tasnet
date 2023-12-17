from pydub import AudioSegment

#E:/Files/code/electromagnetism/tr/s1/

#获取音频时长

def get_duration(inputDir):
    sound = AudioSegment.from_wav(inputDir)
    duration = sound.duration_seconds * 1000  # 音频时长（ms）
    return duration

# 切割函数
def get_wav_make(inputDir,outputDir,start_time,end_time):
    #sound= AudioSegment.from_wav(inputDir)
    #duration = sound.duration_seconds * 1000  # 音频时长（ms）
    duration=get_duration(inputDir)
    begin = start_time
    if(end_time>duration):
        end_time=duration
    end = end_time
    cut_wav = sound[begin:end]   #以毫秒为单位截取[begin, end]区间的音频
    cut_wav.export(outputDir, format='wav')   #存储新的wav文件



sound= AudioSegment.from_wav("E:\\Files\\code\\pretreatment\\electromagnetism\\cv\\mix\\a1.wav")
duration = sound.duration_seconds * 1000  # 音频时长（ms）
print(duration)

'''
get_wav_make("E:\\Files\\code\\pretreatment\\electromagnetism\\cv\\mix\\a1.wav",
             "E:\\Files\\code\\pretreatment\\electromagnetism\\cv\\mix\\split1.wav",
             0,
             5000)
'''

#str1=["tt"]
str1=["tr","cv","tt"]
str2=["s1","s2","mix"]
distance = 10000000
for s1 in str1:
    for s2 in str2:
        path="E:\\Files\\code\\electromagnetism\\"+s1+"\\"+s2+"\\"
        in_path = path + "a1.wav"
        duration=get_duration(in_path)
        begin_time = 0
        end_time = begin_time+distance
        while end_time < duration:
           # in_path = path + "a" + str(num) + ".wav"
            out_path = path + "b_" + str(begin_time) +"_"+str(end_time)+ ".wav"
            get_wav_make(in_path, out_path, begin_time,end_time)
            begin_time = end_time
            end_time = begin_time + distance

print("finish\n")





