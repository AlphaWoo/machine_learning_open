# -*- coding: utf-8 -*-
from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits()
housePrice = datasets.load_boston()
print(digits.images[21])

import matplotlib.pyplot as plt
plt.gray()
plt.matshow(digits.images[22])
plt.show()

print(digits.images[23])
print(digits.images[24])
import matplotlib.pyplot as plt
plt.gray()
plt.matshow(digits.images[25])
plt.show()