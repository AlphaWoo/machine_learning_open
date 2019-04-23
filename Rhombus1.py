# -*- coding: utf-8 -*-
n = 4
_max = 2 * n - 1
a = [("*"*e).center(_max," ")for e in [2*i - 1 if i<=n else 4*n-2*i-1 for i in range(1,_max+1)]]
for each in a:
    print (each)