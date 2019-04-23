# -*- coding: utf-8 -*-
InPut = input('请输入字符串：')
letters  = [ ]
spaces = [ ]
digits   = [ ]
others = [ ] 
for i in iter(InPut):
    if i.isalpha():
        letters.append(i)
    elif i.isspace():
        spaces.append(i)
    elif i.isdigit():
        digits.append(i)
    else:
        others.append(i)
print('''
字母: {}, 个数: {}
空格: {}, 个数: {}
数字: {}, 个数: {}
其他: {}, 个数: {}'''\
.format(letters, len(letters), spaces, len(spaces), digits, len(digits),others, len(others)))