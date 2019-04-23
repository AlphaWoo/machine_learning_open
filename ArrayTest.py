# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 21:09:24 2019

@author: Gary
"""
import numpy as np
A=[1,2,3]
B=[(-1,0,1),(0,0,1),(1,0,2)]
C=[-1,2,-4]
D=np.dot(np.dot(A,B),C)
E=[(-1,0,0),(-1,2,0),(-1,3,5)]
print(np.linalg.inv(E))
print("A=",A)
print("B=",B)
print("C=",C)
print("D=A*B*C",D)