# A PyTorch Implementation of EffiDG


## Requirements

the file `environment.yml` records the requirement packages for this project.

To install the requirement packages, please run the following command:

```sh
conda env create -f environment.yml -n EffiDG
```

Then, the environment can be activated by using the command

```sh
conda activate EffiDG
```


## Datasets

Datasets used in this paper can be found on this [link](http://palm.seu.edu.cn/zhangml/Resources.htm#MIPL_data).



## Demo

To reproduce the results of MNIST_MIPL dataset in the paper, please run the following command:

```sh
python main_graph.py --ds MNIST_MIPL --ds_suffix r1 --bs 350 --lr 5e-4 --epoch 200 --nr_samples 40
python main_graph.py --ds MNIST_MIPL --ds_suffix r2 --bs 350 --lr 5e-4 --epoch 200 --nr_samples 30
python main_graph.py --ds MNIST_MIPL --ds_suffix r3 --bs 350 --lr 5e-4 --epoch 200 --nr_samples 50
```


## Parameter Settings

| Datasets              | Learning Rate | Batch Size | Epochs | nr_sampples   |
| --------------------- | ------------- | ---------- |--------| ------------- |
| MNIST_MIPL (r = 1)    | 0.0005        | 350        | 200    | 40            |
| MNIST_MIPL (r = 2)    | 0.0005        | 350        | 200    | 30            |
| MNIST_MIPL (r = 3)    | 0.0005        | 350        | 200    | 50            |
| FMNIST_MIPL (r = 1)   | 0.0005        | 350        | 200    | 30            |
| FMNIST_MIPL (r = 2)   | 0.0005        | 350        | 200    | 40            |
| FMNIST_MIPL (r = 3)   | 0.0005        | 350        | 200    | 20            |
| Birdsong_MIPL (r = 1) | 0.001         | 910        | 300    | 10            |
| Birdsong_MIPL (r = 2) | 0.002         | 910        | 300    | 20            |
| Birdsong_MIPL (r = 3) | 0.005         | 910        | 300    | 30            |
| SIVAL_MIPL (r = 1)    | 0.002         | 1050       | 300    | 30            |
| SIVAL_MIPL (r = 2)    | 0.005         | 1050       | 300    | 20            |
| SIVAL_MIPL (r = 3)    | 0.005         | 1050       | 300    | 30            |
| CRC-MIPL-Row          | 0.001         | 4900       | 300    | 10            |
| CRC-MIPL-SBN          | 0.001         | 4900       | 300    | 10            |
| CRC-MIPL-KMeansSeg    | 0.001         | 4900       | 300    | 10            |
| CRC-MIPL-SIFT         | 0.001         | 4900       | 300    | 10            |


*N.B.*: This package is only free for academic usage.
