import numpy as np


def getSignalInfornamtion(frame_data,deviation_num):
    standard=(np.max(frame_data)+np.min(frame_data))/2
    # (array([ 4, 11, 18], dtype=int64), array([0, 0, 0], dtype=int64))
    positions_arr = np.where(frame_data < standard)
    #array([ 4, 11, 18], dtype=int64)
    position_arr=positions_arr[0]
    print('position_arr',position_arr)
    #获取开始位置
    start_position=position_arr[0]
    print('start_position', start_position)
    #获取结束位置
    end_position = position_arr[-1]
    print('end_position', end_position)
    # 获取有效长度
    valid_len=position_arr[-1]-position_arr[0]+1
    print(valid_len)

def getSplitSignalPosition(data,standard,start_position,deviation_num):
    i = start_position
    dic=None
    flag=False #flag=True说明查找成功
    while (i < len(data)):
        # 如果当前值不满足标准，则需要看看在误差范围内是否满足
        if (data[i][0] >= standard):
            dic = judgeFollowValue(data, standard, i + 1, deviation_num)
            temp_flag = dic['flag']
            # 在误差范围内不满足
            if (temp_flag == False):
                flag=True
                break
            else:
                i = dic['position']

        i = i + 1

    end_position=i
    if(flag):
        end_position = i-1

    length=end_position-start_position+1

    return length


def judgeFollowValue(data,standard,start_position,deviation_num):
    '''该函数用来预判接下来的num步是否符合标准

    :param data: 判断的数据集
    :param standard: 判断标准
    :param start_position: 开始位置
    :param deviation_num: 判断的个数
    :return: 返回处理字典  flag= True满足标准  position 此时所在的位置
    '''
    i=start_position
    flag=False
    while(deviation_num>0):
        if(data[i][0]<standard):
            flag=True
            break
        i=i+1
        deviation_num=deviation_num-1
    end_position=i
    if(flag==False):
        end_position=i-1
    dic= {'flag': flag, 'position': end_position}
    return dic




'''
def longestDupData(data):
    dic_data={'len': None, 'data': None, 'index': None}
    for i in range(int(len(data) / 2), 0, -1):
        for j in range(int(len(data) / 2)):
            for h in range(j + i, len(data) - i + 1):
                if ((data[j:j + i]==data[h:h + i]).all()) :
                    dic_data['index']=j
                    #print('起始下标：', j)
                    dic_data['len']=i
                    #print('长度：', i)
                    dic_data['data']=data[j:j + i]
                    #print('重复子数组：', data[j:j + i])
                    return dic_data
                    #exit()

'''
#data=np.array([[1],[1],[3],[1],[2],[3],[1],[2],[3]])
data=np.array([[1,2, 3]])
b=data.reshape(3,1)
print(b)

'''
print('len(data)=',len(data),'len(data[0,:])=',len(data[0,:]))


print(data)
for i in range(len(data[0,:])):
    print(data[0,i])
a=data[0,2]
print(data)

'''
'''

#dic=longestDupData(data)
'''
#print('len',dic['len'],'data',dic['data'],'index',dic['index'])