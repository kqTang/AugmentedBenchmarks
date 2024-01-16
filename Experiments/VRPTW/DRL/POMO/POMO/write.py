import xlwt
import torch
import pdb

def write(data):
    xls = xlwt.Workbook()
    sht1 = xls.add_sheet('data')  # epochs,score,loss
    # 设置字体格式
    Font0 = xlwt.Font()
    Font0.name = "Times New Roman"
    Font0.colour_index = 2
    Font0.bold = True  # 加粗
    style0 = xlwt.XFStyle()
    sht1.write(0, 0, 'epochs' ,style0)
    sht1.write(0, 1, 'score' ,style0)
    sht1.write(0, 2, 'loss' ,style0)
    for i in range(len(data['epoch'])):
        sht1.write(i + 1, 0, data['epoch'][i], style0)  # 顾客点的坐标x与y
        sht1.write(i + 1, 1, data['train_score'][i], style0)
        sht1.write(i + 1, 2, data['train_loss'][i], style0)  # 顾客点的需求
    xls.save('./xls/hard/rc2.xls')
data = torch.load('./pt/hard/rc2/result-3000.pth')
write(data)