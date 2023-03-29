## Log Sequence Anomaly Detection

## Use Case
Identify log anomalies from sequence of logs generated dataset.

### Version
1.0

### Model Overview
The model is a sequence binary classifier trained with vector representation of log sequence of BGL dataset. The task is to identify abnormal log sequence of alerts from sequence of normally generated logs. This work is based on the model developed in the works of [[2](https://ieeexplore.ieee.org/document/9671642),[3](https://github.com/hanxiao0607/InterpretableSAD)], for further detail refer the paper and associated code at the reference link.

### Model Architecture
LSTM binary classifier with word2vector embedding input. 

### Requirements

Requirements can be installed with 
```
pip install -r requirements.txt
```

### Training

#### Training data
The dataset for the example used from BlueGene/L Supercomputer System (BGL). BGL dataset contains 4,747,963 log messages that are collected
from a [BlueGeme/L]('https://zenodo.org/record/3227177/files/BGL.tar.gz?download=1') supercomputer system at Lawrence Livermore National Labs. The log messages can be categorized into alert and not-alert messages. The log message is parsed using [Drain](https://github.com/logpai/logparser) parser for preprocessing. The model is trained and evaluated using 1 million rows of preprocessed logs. For running the workflow a preprocessed smaller set of BGL can be used from https://github.com/LogIntelligence/LogPPT

#### Training parameters

For the Word2Vec, gensim model is used with size=8
Parameter for the LSTM model.
```
output_dim = 2
emb_dim = 8
hidden_dim = 128
n_layers = 1
dropout = 0.0
batch_size = 32
n_epoch = 10
```
#### GPU Model
Tesla V100-SXM2

#### Model accuracy
The label distribution in the dataset is imbalanced, the F1 score over the 1 million row dataset is 0.97.


#### Training script
To train the model, run the code in the notebook. This will save trained model under `model` directory.

### Inference
To run inference from trained model 
```bash
python inference.py --model_name model/model_BGL.pt --input_data dataset/BGL_2k.log_structured.csv

```
This will produce `result.csv` that contains binary prediction of the model.

### How To Use This Model
This model is an example of sequence binary classifier. This model requires parsed log messages as input for training and inference. The model and Word2Vector embedding is trained as follows in the training notebook. During inference, trained model is loaded from `model` directory and input file in the form of parsed logs are expected to output prediction for sequences of log messages.

### Input
The input is an output of parsed system log messages represented as CSV file.

### Output
Binary classifier output assigned to each sequence log messages in the input file. The predicted output is appended at the last column of the input sequence.

#### Out-of-scope use cases
N/A

### Ethical considerations
N/A

### Reference
1. https://arxiv.org/pdf/2202.04301.pdf
2. https://ieeexplore.ieee.org/document/9671642
3. https://github.com/hanxiao0607/InterpretableSAD