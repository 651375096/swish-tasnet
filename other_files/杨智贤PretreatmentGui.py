from tkinter import *
import threading#多线程
from tkinter import filedialog
import 本地GUI陈扬按行信号cls预处理wav复现 as pre
from tkinter import ttk #下拉控件
import tkinter as tk
import pdb
# -----------------------窗口----------------------------
window = Tk()
window.title("这是一个窗口")
window.geometry("1200x550")
# window.overrideredirect(1)#删除标题栏
window.resizable(0,0) #禁止拉伸

#定义画布
canvas=Canvas(window,width=600,height=600)
canvas.place(x=0,y=0)

canvas1=Canvas(window,width=600,height=600)
canvas1.place(x=600,y=0)
window.mainloop()
pdb.set_trace()
#--------------------------------------------------------
# ----------------按键逻辑初始化--------------------------
# 输入
group1_click=0
group2_click=0
# 输出路径
swav_click = 0
stxt_click = 0
# 参数调整
stander_click = 0
mask_click = 0
# 确定/退出
exit_click=0
confirm_click=0
flag= 0
#选择group1模块
def group1():
    global group1_click
    group1_click+=1


#选择group2模块
def group2():
    global group2_click
    group2_click+=1
#生成路径(wav)
def subDirwav():
    global swav_click
    swav_click +=1
#生成路径(txt)
def subDirtxt():
    global stxt_click
    stxt_click +=1
# 确认
def confirm():
    global confirm_click
    confirm_click += 1
# 二值化
def Binarize():
    global stander_click
    stander_click +=1
# 掩码替换
def Mask():
    global mask_click
    mask_click += 1
#退出
def q():
    global exit_click
    exit_click+=1

#--------------------------------------------------------
#-----------------------功能模块--------------------------
def justify():

    def cc():
        global group1_click,group2_click,exit_click,swav_click,stxt_click,confirm_click
        global selected_group2_files_path,subDir_wav,subDIr_txt,flag,stander_content,replace_point,point_num,aim_value
        global stander_click,mask_click
        # 设置二值化标准赋值
        def func1(event):
            global stander_content
            stander_content = cbox.get()
        # 设置掩码替换
        def func2():
            global replace_point
            replace_point = entry01.get()

        def func3():
            global point_num
            point_num = entry02.get()

        def func4():
            global aim_value
            aim_value = entry03.get()

        while True:

            # 选取group1
            if group1_click:
                group1_click = 0
                selected_group1_files_path = filedialog.askopenfilenames()  # askopenfilenames函数选择多个文件
                selected_group1_files_path = list(selected_group1_files_path)

                group1_txt = Text(window,width=27, height=20)
                a = str(selected_group1_files_path) #将列表中的空格换位换行
                a = a.replace(" ","\n\n")
                group1_txt.insert(INSERT,a)
                group1_txt.place(x = 0,y =200)


            # 选取group2
            if group2_click:
                group2_click = 0
                selected_group2_files_path = filedialog.askopenfilenames()  # askopenfilenames函数选择多个文件
                selected_group2_files_path = list(selected_group2_files_path)

                group2_txt = Text(window,width=27, height=20)
                b = str(selected_group1_files_path)#将列表中的空格换位换行
                b = b.replace(" ","\n\n")
                group2_txt.insert(INSERT,b)
                group2_txt.place(x = 200,y =200)


            # 选择wav生成路径
            if swav_click:
                swav_click = 0
                subDir_wav = filedialog.askdirectory()
                subDir_wav = subDir_wav+"/"
                group3_txt = Text(window, width=27, height=20)
                group3_txt.insert(INSERT, subDir_wav)
                group3_txt.place(x=400, y=200)
            # 选择txt生成路径
            if stxt_click:
                stxt_click = 0
                subDIr_txt = filedialog.askdirectory()
                subDIr_txt = subDIr_txt+"/"
                group4_txt = Text(window, width=27, height=20)
                group4_txt.insert(INSERT, subDIr_txt)
                group4_txt.place(x=600, y=200)
            # 设置二值化参数
            if stander_click:
                stander_click = 0
                # 二值化标准下拉框
                cbox = ttk.Combobox(window)
                cbox.place(x=0, y=50)
                cbox['value'] = ('average', 'range', 'min')
                cbox.current(0)
                # print(cbox.get() + "\n")
                cbox.bind("<<ComboboxSelected>>", func1)
            #  设置掩码替换参数
            if mask_click:
                mask_click = 0

                label1 = tk.Label(window, text="定位点的值：")
                label1.place(x=200,y=50)
                v1 = tk.StringVar()
                entry01 = tk.Entry(window, textvariable=v1,width = 10)
                entry01.place(x=280, y=50)

                label2 = tk.Label(window, text="替换的数量：")
                label2.place(x=200, y=100)
                v2 = tk.StringVar()
                entry02 = tk.Entry(window, textvariable=v2,width = 10)
                entry02.place(x=280, y=100)

                label2 = tk.Label(window, text="替换的值：")
                label2.place(x=200, y=150)
                v3 = tk.StringVar()
                entry03 = tk.Entry(window, textvariable=v3,width = 10)
                entry03.place(x=280, y=150)


            # 确定
            if confirm_click:
                confirm_click=0
                flag = 1
            # 退出
            if exit_click:
                exit_click = 0
                window.destroy()
            # 运行程序
            if flag:
                flag = 0
                func2()
                func3()
                func4()

                pre.getDataset1(selected_group1_files_path,selected_group2_files_path,subDir_wav,subDIr_txt,stander_content,replace_point,point_num,aim_value)

    threading.Thread(target=cc).start()




justify()



# ----------------按钮界面设置----------------------------



bt_stander= Button(window,text='选择二值化标准',height=2,width=27,command=Binarize)
bt_stander.place(x=0,y=0)

bt_stander= Button(window,text='设置掩码替换参数',height=2,width=27,command=Mask)
bt_stander.place(x=200,y=0)

bt_i= Button(window,text='选择group1',height=2,width=27,command=group1)
bt_i.place(x=0,y=500)

bt_o = Button(window,text='选择group2',height=2,width=27,command=group2)
bt_o.place(x=200,y=500)

bt_c = Button(window,text='选择wav输出路径',height=2,width=27,command=subDirwav)
bt_c.place(x=400,y=500)

bt_k = Button(window,text='选择txt输出路径',height=2,width=27,command=subDirtxt)
bt_k.place(x=600,y=500)

bt_r = Button(window,text='确认',height=2,width=27,command=confirm)
bt_r.place(x=800,y=500)

bt_q = Button(window,text='退出',height=2,width=27,command=q)
bt_q.place(x=1000,y=500)

window.mainloop()

#--------------------------------------------------------