# Towards Spatio-Temporal Aware Traffic Time Series Forecasting


This is a PyTorch implementation of EnhanceNet in the following paper: \
Razvan-Gabriel Cirstea, Tung Kieu, Chenjuan Guo, Shirui Pan, Bin Yang. Towards Spatio-Temporal Aware Traffic Time Series Forecasting.


## Requirements
* torch
* scipy>=0.19.0
* numpy>=1.12.1
* pandas>=0.19.2
* pyyaml
* statsmodels
* torch
* tables
* future

Dependency can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Data Preparation
The traffic data files [here](https://github.com/Davidham3/STSGCN). 

## Run the Model 

To train the model on different datasets just use the command:

```
python train.py 
```

By default it will run the experiments on PEMS4 dataset. 
To select another dataset open run.py and modify DATASET = 'PEMSX' 
where X is one of the datasets [3,4,7,8]. 

The configurations file are located in the config directory. For changing any of the hyper-parameters modify the conf file 
associated with the dataset and rerun the above command.

