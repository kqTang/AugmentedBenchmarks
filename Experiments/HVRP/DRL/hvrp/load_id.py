import re
import numpy as np
import pdb
import torch
import argparse
import pickle
def load_Choia_Tcha(instance_id):
    Dataset = []
    filename = 'data/hcvrp/VFMP_FV/vfmpfv{:02d}.txt'.format(instance_id)
    print(filename)
    dataset = {
    "veh": {},
    "loc": {},
    "depot":{},
    "q": {}
}   
    size={
        3:20,4:20,5:20,6:20,
        13:50,14:50,15:50,16:50,
        17:75,18:75,
        19:100,20:100
    }.get(instance_id,None)
    with open(filename, encoding='ISO-8859-15') as f:
        content = f.readlines()
        '''Vehicle'''
        type_veh = int(re.split('\s+', content[5].strip())[1])
        
        # Find lines with vehicle information and handle potential float values
        vehicles_info = []
        for line in content[6:6+type_veh]:
            # Split line into items and convert to float to avoid ValueError, then to int
            line_items = map(float, line.split())
            vehicles_info.append([int(item) if item.is_integer() else float(item) for item in line_items])
        dataset['veh'] = vehicles_info
        '''Customer'''
        dataset['depot'] = [int(i) for i in content[6+type_veh].split()]
        customer = [[int(i) for i in line.split()] for line in content[7+type_veh : 7+type_veh+size+1]]
        dataset['loc'] = [ i[:-1] for i in customer if i !=[]]
        dataset['q'] =[ i[-1] for i in customer  if i !=[]]
        Dataset.append(dataset)
        print(Dataset)
    with open('data/hcvrp/hvrp_{}.pkl'.format(instance_id), 'wb') as f:
        pickle.dump(Dataset, f, 5)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    opts = parser.parse_args()
    for i in [3,4,5,6,13,14,15,16,17,18,19,20]:
        load_Choia_Tcha(i)