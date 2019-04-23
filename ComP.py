# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 22:23:33 2019

@author: lenovo
"""

import numpy as np
a=eval(input("请输入第一个数a："))
b=eval(input("请输入第一个数b："))
c=eval(input("请输入第一个数c："))

if a>=b:
    if b>=c:
        print("a,b,c的大小顺序为：a>b>c")
    elif a>=c:
        print("a,b,c的大小顺序为：a>c>b")
    else:
        print("a,b,c的大小顺序为：c>a>b")
elif b>a:
    if a>=c:
        print("a,b,c的大小顺序为：b>a>c")
    elif c>=b:
        print("a,b,c的大小顺序为：c>b>a")
    else:
        print("a,b,c的大小顺序为：b>c>a")