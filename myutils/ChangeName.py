import os
#E:/Files/学习/毕业设计/电磁信号分离整理/electromagnetism/cv/s2/
path = input('请输入文件路径(结尾加上/)：')

# 获取该目录下所有文件，存入列表中
f = os.listdir(path)
print(f)
'''
n = 0
for i in f:
    # 设置旧文件名（就是路径+文件名）
    oldname = path + f[n]

    # 设置新文件名
    newname = path + 'a' + str(n + 1) + '.wav'

    # 用os模块中的rename方法对文件改名
    os.rename(oldname, newname)
    print(oldname, '======>', newname)

    n += 1

'''