# -*- coding: utf-8 -*-
def add_dic(dic):
    while True:
        word=input("请输入英文单词（直接按回车结束）：")
        if len(word)==0:
            break;
        meaning=input("请输入中文翻译：")
        dic[word]=meaning
        print("该单词已添加到字典库。")
    return
def search_dic(dic):
    while True:
        word=input("请输入要查询的英文单词（直接按回车结束）：")
        if len(word)==0:
            break;
        if word in dic:
            print("%s的中文翻译是%s"%(word,dic[word]))
        else:
            print("字典库中未找到这个单词")
    return
  
worddic=dict()
while True:
    print("请选择功能：\n1:输入\n2：查找\n3：退出")
    c=input()
    if c=="1":
        add_dic(worddic)
    elif c=="2":
        search_dic(worddic)
    elif c=="3":
        break
    else:
        print("输入有误！")
