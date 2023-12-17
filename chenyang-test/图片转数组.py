import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image = Image.open('../test.png')
plt.imshow(image)
plt.show()
image = np.copy(image)  # 这一句
print(image)
print(image.shape)
