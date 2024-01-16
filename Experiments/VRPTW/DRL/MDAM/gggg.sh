#!/bin/bash

for (( num = 80; num <= 98; num++ )) ;
do
   python run.py --train_type c1 --load_path="pretrained/cvrptw/c1/epoch-$num.pt" --val_dataset ./data/cvrptw/cvrptw_c1_test_seed4321.pkl --graph_size 100 --baseline rollout --run_name 'c1' --problem="cvrptw" --kl_loss=0.01 --n_EG=2 --n_paths=5 --val_size=1024 --eval_only
done
