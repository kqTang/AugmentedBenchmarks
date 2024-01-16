# POMO_2_CVRPTW
## Dependencies

These are ready on many cloud servers.
* Python>=3.8
* pytorch<2.0
* pytz
* matplotlib
* tqdm

## Notice

All commands need to be executed in `./POMO`

## Generate DER-Solomon data
### Data for testing
```bash
python generate_data.py --train_type=c1 --num_samples=1024 --size=100 --trainortest='test' --solomon_train
```
### Data for training
```bash
python generate_data.py --train_type=c1 --size=100 --trainortest='train' --solomon_train
```
## Train
Set --batch_size according to your GPU memory, and must to be divided evently by generated data size:
```bash
python train.py --size=100 --batch_size=128 --dataname=./pth/data_c1_train.pth --hard
```
## Test
### Test for 1024 generated instances of DER-Solomon
Test for series c1 of DER-Solomon as an example, the trained model is in ./pt/hard:
```bash
python test.py --size 100 --hard --batch_size=1024 --model_load_path ./pt/hard/c1/checkpoint-3000.pt --train_type c1  --filename ./pth/data_c1_test.pth 
```
### Test for Solomon benchmark
Must set --solomon. Test for each series:
```bash
python test.py --hard --solomon --model_load_path ./pt/hard/c1/checkpoint-3000.pt --train_type c1 --file_pre ./SolomonBenchmark/C1/ --filename Solomon1.txt Solomon2.txt Solomon3.txt Solomon4.txt Solomon5.txt Solomon6.txt Solomon7.txt Solomon8.txt Solomon9.txt
python test.py --hard --solomon --model_load_path ./pt/hard/c2/checkpoint-3000.pt --train_type c2 --file_pre ./SolomonBenchmark/C2/ --filename Solomon1.txt Solomon2.txt Solomon3.txt Solomon4.txt Solomon5.txt Solomon6.txt Solomon7.txt Solomon8.txt
python test.py --hard --solomon --model_load_path ./pt/hard/c2/checkpoint-3000.pt --train_type r1 --file_pre ./SolomonBenchmark/C2/ --filename Solomon1.txt Solomon2.txt Solomon3.txt Solomon4.txt Solomon5.txt Solomon6.txt Solomon7.txt Solomon8.txt Solomon9.txt Solomon10.txt Solomon11.txt Solomon12.txt
python test.py --hard --solomon --model_load_path ./pt/hard/c2/checkpoint-3000.pt --train_type r2 --file_pre ./SolomonBenchmark/C2/ --filename Solomon1.txt Solomon2.txt Solomon3.txt Solomon4.txt Solomon5.txt Solomon6.txt Solomon7.txt Solomon8.txt Solomon9.txt Solomon10.txt Solomon11.txt
python test.py --hard --solomon --model_load_path ./pt/hard/c2/checkpoint-3000.pt --train_type rc1 --file_pre ./SolomonBenchmark/C2/ --filename Solomon1.txt Solomon2.txt Solomon3.txt Solomon4.txt Solomon5.txt Solomon6.txt Solomon7.txt Solomon8.txt
python test.py --hard --solomon --model_load_path ./pt/hard/c2/checkpoint-3000.pt --train_type rc2 --file_pre ./SolomonBenchmark/C2/ --filename Solomon1.txt Solomon2.txt Solomon3.txt Solomon4.txt Solomon5.txt Solomon6.txt Solomon7.txt Solomon8.txt
```

# The next is original POMO, as .py version

We provide codes for two CO (combinatorial optimization) problems:<br>
- Traveling Salesman Problem (TSP) <br>
- Capacitated Vehicle Routing Problem (CVRP) <br>


### Changes from the old version

Other than the re-organized structure, no major change has been made, so that the two versions should give roughly the same results.
Some meaningful changes are:
- Query token in the decoder does not contain "graph encoding" now, because this does not seem to make much difference. (But who knows?)
- Normalization methods have changed, from BatchNorm to InstanceNorm. (It seemed more natural. But this may have affected the model's performance in a negative way.) 


### Basic Usage

To train a model, run *train.py*. <br>
*train.py* contains parameters you can modify. At the moment, it is set to train N=20 problems. <br>
<br>
To test a model, run *test.py*. <br>
You can specify the model as a parameter contained in *test.py*. At the moment, it is set to use the saved model (N=20) we have provided (in *result* folder), but you can easily use the one you have trained using *train.py*.


### Used Libraries
python v3.7.6 <br>
torch==1.7.0 <br>