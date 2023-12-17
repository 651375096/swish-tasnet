import os
import re

folder = './exp/tmp/out/signal/estimate/s1'

# 遍历文件夹获取所有txt文件名
txt_files = [f for f in os.listdir(folder) if f.endswith('.txt')]
# 排序文件名
txt_files.sort()
# 获取汇总文件名(取所有文件名的公共部分)
common_name = os.path.commonprefix(txt_files)
all_file = common_name + 'all.txt'

with open(all_file, 'w') as f:
    for file in txt_files:
        with open(os.path.join(folder, file)) as f2:
            f.write(f2.read() + '\n')

print('汇总文件已生成:', all_file)