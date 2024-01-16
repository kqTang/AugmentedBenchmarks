import xlwt
import torch
import pdb


def write(costs, tours):
    xls = xlwt.Workbook()
    sht1 = xls.add_sheet('costs_tours')
    # 设置字体格式
    Font0 = xlwt.Font()
    Font0.name = "Times New Roman"
    Font0.colour_index = 2
    Font0.bold = True  # 加粗
    style0 = xlwt.XFStyle()
    # sht1.write(0, 0, 'epochs', style0)
    # sht1.write(0, 1, 'score', style0)
    # sht1.write(0, 2, 'loss', style0)
    for i in range(costs.shape[0]):
        sht1.write(0, i+2, costs[i].item(), style0)
        for j in range(len(tours[i])):
            sht1.write(j + 1, i+2, tours[i][j].item() + 1, style0)
    xls.save('./xls/costs_tours.xls')
