import xlwt
import xlrd
import torch
import pdb


def write(data,line):
    # xlrd.open_workbook("temp.xls")
    xls = xlrd.open_workbook("temp.xls")
    sht1 = xls.sheets()[0]  # epochs,score,loss
    # 设置字体格式
    Font0 = xlwt.Font()
    Font0.name = "Times New Roman"
    Font0.colour_index = 2
    Font0.bold = True  # 加粗
    style0 = xlwt.XFStyle()
    print(data.size(0))
    for i in range(data.size(0)):
        sht1.write(line, i, data[i].item(), style0)  
    xls.save('temp.xls')
