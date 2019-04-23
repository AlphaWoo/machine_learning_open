# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
path1= '/home/chenze/MSTdata/xh-mst20120424'
path=path1+'/XHT_MST01_DJH_L11_STP_20120424001542.dat'
data = pd.read_table(path,header=None,skiprows=[0,1],sep='\s+')
#skiprows[a,b,c,d] abcd行不读取//// sep=‘\s+’识别切割的字符（空格，或多个空格），默认为 “，”。   
#lens=len(data)
print(data)
da=data.values[0]
#print(da)
#print(len(da))     
path=path1+'/XHT_MST01_DJH_L11_STP_20120424014536.txt'
data1 = pd.read_csv(path,header=None,skiprows=[0,1],sep='\s+')
lens=len(data1)
#print(data1)

