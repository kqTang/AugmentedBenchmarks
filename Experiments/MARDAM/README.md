# MARDAM

## Install LKH and OR-Tools at first.

LKH:
```bash
tar -zxvf LKH-3.0.5.tar.gz
cd LKH-3.0.5
make
```
OR-Tools:
```bash
python -m pip install --upgrade --user ortools
```
## Training

Training use DER-solomon data:

```bash
CUDA_VISIBLE_DEVICES=0 python ./script/train.py -n=100 -m=25 --use_solomon_train --train_type='all' --LKH_ENABLED
```

Training use origin MARDAM data:

```bash
CUDA_VISIBLE_DEVICES=1 python ./script/train.py -n=100 -m=25 --LKH_ENABLED
```

## Test with Solomon benchmark

Using model trained with origin training dataset:

```bash
CUDA_VISIBLE_DEVICES=3 python ./script/test.py -n=100 -m=25 --load_path ./output/MARDAM_100_230714-1643/chkpt_ep100.pyth
```

Using model trained with DER-Solomon dataset:

```bash
CUDA_VISIBLE_DEVICES=3 python ./script/test.py -n=100 -m=25 --load_path ./output/MARDAM_solomon_230714-1523/chkpt_ep100.pyth
```

Any arguments, like `customer size, problem, batch size, etc.` , can be found in `./utils/_args.py`

The record cost of each epoch is in `./output/loss_gap.csv`

# LKH & OR-Tools

## Solve Solomon

LKH:

```bash
python ./script/train.py -n=100 -m=25 --use_tool_to_test_solomon --LKH_ENABLED
```

OR-Tools:

```bash
python ./script/train.py -n=100 -m=25 --use_tool_to_test_solomon 
```

original author:

 https://gitlab.inria.fr/gbono/mardam

```bash
@article{bono2020solving,
  title={Solving multi-agent routing problems using deep attention mechanisms},
  author={Bono, Guillaume and Dibangoye, Jilles S and Simonin, Olivier and Matignon, La{\"e}titia and Pereyron, Florian},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={22},
  number={12},
  pages={7804--7813},
  year={2020},
  publisher={IEEE}
}
```