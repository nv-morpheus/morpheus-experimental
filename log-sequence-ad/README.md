## Log Sequence Anomaly Detection

## Use Case
Identify log anomalies from sequence of logs generated dataset.

### Version
1.0

### Model Overview
The model is a sequence binary classifier trained with vector representation of log sequence of BGL dataset. The task is to identify abnormal log sequence of alerts from sequence of normally generated logs.

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

### Ethical considerations
N/A

### Reference
1. https://arxiv.org/pdf/2202.04301.pdf
2. https://ieeexplore.ieee.org/document/9671642