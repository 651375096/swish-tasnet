import os



def removeSameFile(filePath1,filePath2,filePath3):
    '''该函数用来删除三个文件夹内的不同文件

    :param filePath1:
    :param filePath2:
    :param filePath3:
    :return:
    '''

    #获取文件内的文件名
    list1 = os.listdir(filePath1)
    list2 = os.listdir(filePath2)
    list3 = os.listdir(filePath3)

    #找到两个文件夹相同的文件
    two_folders_same = [x for x in list2 if x in list1]
    #找到三个文件件的相同文件
    three_folders_same = [x for x in list3 if x in two_folders_same]
    #根据相同的文件，推出每个文件夹中不同的文件
    folder1_diff = [y for y in list1 if y not in three_folders_same]
    folder2_diff = [y for y in list2 if y not in three_folders_same]
    folder3_diff = [y for y in list3 if y not in three_folders_same]
    # 然后将不同文件删除
    for i in folder1_diff:
        os.remove(os.path.join(filePath1, i))
    for i in folder2_diff:
        os.remove(os.path.join(filePath2, i))
    for i in folder3_diff:
        os.remove(os.path.join(filePath3, i))

    remove_file={"folder1":folder1_diff,"folder2":folder2_diff,"folder3":folder3_diff}
    return remove_file

filePath1='E:/Files/folder1/'
filePath2='E:/Files/folder2/'
filePath3='E:/Files/folder3/'
print(removeSameFile(filePath1,filePath2,filePath3))