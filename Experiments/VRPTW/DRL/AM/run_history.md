CUDA_VISIBLE_DEVICES=9 python eval.py data/cvrptw/cvrptw_c2_test_seed4321.pkl --model pretrained/cvrptw/c2 --decode_strategy sample --width 1024 --eval_batch_size 8 -f

CUDA_VISIBLE_DEVICES=9 python eval.py data/cvrptw/cvrptw_c1_test_seed4321.pkl --model pretrained/cvrptw/c1 --decode_strategy sample --width 1024 --eval_batch_size 9 -f

CUDA_VISIBLE_DEVICES=9 python eval.py data/cvrptw/cvrptw_r1_test_seed4321.pkl --model pretrained/cvrptw/r1 --decode_strategy sample --width 1024 --eval_batch_size 12 -f

CUDA_VISIBLE_DEVICES=9 python eval.py data/cvrptw/cvrptw_r2_test_seed4321.pkl --model pretrained/cvrptw/r2 --decode_strategy sample --width 1024 --eval_batch_size 11 -f

CUDA_VISIBLE_DEVICES=9 python eval.py data/cvrptw/cvrptw_rc1_test_seed4321.pkl --model pretrained/cvrptw/rc1 --decode_strategy sample --width 1024 --eval_batch_size 8 -f

CUDA_VISIBLE_DEVICES=9 python eval.py data/cvrptw/cvrptw_rc2_test_seed4321.pkl --model pretrained/cvrptw/rc2 --decode_strategy sample --width 1024 --eval_batch_size 8 -f
