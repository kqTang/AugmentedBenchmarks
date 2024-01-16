# Generate benchmark to .pkl
 python load_id.py
# Test Golden benchmark
echo "Test Golden benchmark"
CUDA_VISIBLE_DEVICES=3 python eval.py data/hcvrp/hvrp_3.pkl --instances Gloden --model pretrained/old/Gloden/20_5/epoch-33.pt --decode_strategy sample --width 12800 --eval_batch_size 1 -f
CUDA_VISIBLE_DEVICES=3 python eval.py data/hcvrp/hvrp_4.pkl --instances Gloden --model pretrained/old/Gloden/20_3/epoch-15.pt --decode_strategy sample --width 12800 --eval_batch_size 1 -f
CUDA_VISIBLE_DEVICES=3 python eval.py data/hcvrp/hvrp_5.pkl --instances Gloden --model pretrained/old/Gloden/20_5/epoch-33.pt --decode_strategy sample --width 12800 --eval_batch_size 1 -f
CUDA_VISIBLE_DEVICES=3 python eval.py data/hcvrp/hvrp_6.pkl --instances Gloden --model pretrained/old/Gloden/20_3/epoch-15.pt --decode_strategy sample --width 12800 --eval_batch_size 1 -f
CUDA_VISIBLE_DEVICES=3 python eval.py data/hcvrp/hvrp_13.pkl --instances Gloden --model pretrained/old/Gloden/50_6/epoch-54.pt --decode_strategy sample --width 12800 --eval_batch_size 1 -f
CUDA_VISIBLE_DEVICES=3 python eval.py data/hcvrp/hvrp_14.pkl --instances Gloden --model pretrained/old/Gloden/50_3_14/epoch-2.pt --decode_strategy sample --width 12800 --eval_batch_size 1 -f
CUDA_VISIBLE_DEVICES=3 python eval.py data/hcvrp/hvrp_15.pkl --instances Gloden --model pretrained/old/Gloden/50_3_15/epoch-48.pt --decode_strategy sample --width 12800 --eval_batch_size 1 -f
CUDA_VISIBLE_DEVICES=3 python eval.py data/hcvrp/hvrp_16.pkl --instances Gloden --model pretrained/old/Gloden/50_3_15/epoch-48.pt --decode_strategy sample --width 12800 --eval_batch_size 1 -f
CUDA_VISIBLE_DEVICES=3 python eval.py data/hcvrp/hvrp_17.pkl --instances Gloden --model pretrained/old/Gloden/75_4/epoch-10.pt --decode_strategy sample --width 12800 --eval_batch_size 1 -f
CUDA_VISIBLE_DEVICES=3 python eval.py data/hcvrp/hvrp_18.pkl --instances Gloden --model pretrained/old/Gloden/75_6/epoch-24.pt --decode_strategy sample --width 12800 --eval_batch_size 1 -f
CUDA_VISIBLE_DEVICES=3 python eval.py data/hcvrp/hvrp_19.pkl --instances Gloden --model pretrained/old/Gloden/100_3_19/epoch-5.pt --decode_strategy sample --width 12800 --eval_batch_size 1 -f
CUDA_VISIBLE_DEVICES=3 python eval.py data/hcvrp/hvrp_20.pkl --instances Gloden --model pretrained/old/Gloden/100_3_20/epoch-47.pt --decode_strategy sample --width 12800 --eval_batch_size 1 -f

# Test Choia&Tcha benchmark
echo "Test Choia&Tcha benchmark"
CUDA_VISIBLE_DEVICES=3 python eval.py data/hcvrp/hvrp_3.pkl --instances Choia\&Tcha --model pretrained/old/Choia/20_5/epoch-44.pt --decode_strategy sample --width 12800 --eval_batch_size 1 -f
CUDA_VISIBLE_DEVICES=3 python eval.py data/hcvrp/hvrp_4.pkl --instances Choia\&Tcha --model pretrained/old/Choia/20_3/epoch-45.pt --decode_strategy sample --width 12800 --eval_batch_size 1 -f
CUDA_VISIBLE_DEVICES=3 python eval.py data/hcvrp/hvrp_5.pkl --instances Choia\&Tcha --model pretrained/old/Choia/20_5/epoch-44.pt --decode_strategy sample --width 12800 --eval_batch_size 1 -f
CUDA_VISIBLE_DEVICES=3 python eval.py data/hcvrp/hvrp_6.pkl --instances Choia\&Tcha --model pretrained/old/Choia/20_3/epoch-45.pt --decode_strategy sample --width 12800 --eval_batch_size 1 -f
CUDA_VISIBLE_DEVICES=3 python eval.py data/hcvrp/hvrp_13.pkl --instances Choia\&Tcha --model pretrained/old/Choia/50_6/epoch-34.pt --decode_strategy sample --width 12800 --eval_batch_size 1 -f
CUDA_VISIBLE_DEVICES=3 python eval.py data/hcvrp/hvrp_14.pkl --instances Choia\&Tcha --model pretrained/old/Choia/50_3/epoch-48.pt --decode_strategy sample --width 12800 --eval_batch_size 1 -f
CUDA_VISIBLE_DEVICES=3 python eval.py data/hcvrp/hvrp_15.pkl --instances Choia\&Tcha --model pretrained/old/Choia/50_3/epoch-48.pt --decode_strategy sample --width 12800 --eval_batch_size 1 -f
CUDA_VISIBLE_DEVICES=3 python eval.py data/hcvrp/hvrp_16.pkl --instances Choia\&Tcha --model pretrained/old/Choia/50_3/epoch-48.pt --decode_strategy sample --width 12800 --eval_batch_size 1 -f
CUDA_VISIBLE_DEVICES=3 python eval.py data/hcvrp/hvrp_17.pkl --instances Choia\&Tcha --model pretrained/old/Choia/75_4/epoch-32.pt --decode_strategy sample --width 12800 --eval_batch_size 1 -f
CUDA_VISIBLE_DEVICES=3 python eval.py data/hcvrp/hvrp_18.pkl --instances Choia\&Tcha --model pretrained/old/Choia/75_6/epoch-30.pt --decode_strategy sample --width 12800 --eval_batch_size 1 -f
CUDA_VISIBLE_DEVICES=3 python eval.py data/hcvrp/hvrp_19.pkl --instances Choia\&Tcha --model pretrained/old/Choia/100_3/epoch-36.pt --decode_strategy sample --width 12800 --eval_batch_size 1 -f
CUDA_VISIBLE_DEVICES=3 python eval.py data/hcvrp/hvrp_20.pkl --instances Choia\&Tcha --model pretrained/old/Choia/100_3/epoch-36.pt --decode_strategy sample --width 12800 --eval_batch_size 1 -f