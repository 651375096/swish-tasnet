import os
import shutil

def copyDir(original,target):
    '''将一个目录下的全部文件和目录,完整地<拷贝并覆盖>到另一个目录


    :param original: 源目录
    :param target: 目标目录
    :return:
    '''

    if not os.path.exists(original):
        # 如果传进来的不是目录
        print("传入的原文件夹路径不存在")
        return

    if not os.path.exists(target):
        os.makedirs(target)  # 创建路径

    for a in os.walk(original):
        #递归创建目录
        for d in a[1]:
            dir_path = os.path.join(a[0].replace(original,target),d)
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)
        #递归拷贝文件
        for f in a[2]:
            dep_path = os.path.join(a[0],f)
            arr_path = os.path.join(a[0].replace(original,target),f)
            shutil.copy(dep_path,arr_path)


#mergeTwoFolders("./data","./data","")

copyDir("./1","./data/3")
copyDir("./2","./data/3")