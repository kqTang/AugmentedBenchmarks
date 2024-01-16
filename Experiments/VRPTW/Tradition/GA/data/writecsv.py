# 导包
import csv
import torch
import pdb
from torch.utils.data import DataLoader
from tqdm import tqdm
# 创建或打开文件
csvfile = open('./data_for_GA/cvrptw_c1_testDER_seed4321.csv', mode='w', newline='')
# 标题列表
fieldnames = ['x', 'y', 'demand', 'e', 'l','service_time']
data=torch.load('./pth/data_c1_2_test_200.pth')['data']
# 创建 DictWriter 对象

write = csv.DictWriter(csvfile, fieldnames=fieldnames)
# 写入表头
write.writeheader()
# 写入数据
for batch in DataLoader(data, batch_size=1000):
    # pdb.set_trace()
    write.writerow({'x': batch['depot'][0, 0].item(), 'y': batch['depot'][0, 1].item(), 'demand': 0,
                    'e': 0, 'l': batch['leave_time'][0, 0].item(), 'service_time': 0})
    for i in tqdm(range(1000)):
        for j in range (batch['loc'].shape[1]):
            write.writerow({'x': batch['loc'][i, j, 0].item(), 'y': batch['loc'][i, j, 1].item(), 'demand': int(batch['demand'][i, j].item()*700),
                            'e': batch['enter_time'][i, j+1].item(), 'l': int(batch['leave_time'][i, j+1].item()), 'service_time': batch['service_duration'][i, j].item()})


