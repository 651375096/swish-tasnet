# import matplotlib.pyplot as plt # plt 用于显示图片
# import matplotlib.image as mpimg # mpimg 用于读取图片
# import numpy as np
# from PIL import Image
# image = Image.open('../test.png')
# # img 是array
# plt.imshow(image) # 显示图片
# plt.show()
# # 如果img的取值范围是【0，255】，那么用下面的语句
# # plt.imshow(out.astype('uint8'))
# # 保存图片
# # plt.savefig('img.jpg')

import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np

# img 是array
plt.imshow(img) # 显示图片
# 如果img的取值范围是【0，255】，那么用下面的语句
plt.imshow(out.astype('uint8'))
# 保存图片
plt.savefig('img.jpg')
