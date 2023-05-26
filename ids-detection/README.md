## Intrusion Detection Using Lightweight Online Detector of Anomalies (LODA)

## Use Case
Intrusion detection using Lightweight Online Detector of Anomalies (LODA)

### Version
1.0

### Model Overview
The model is  a Loda anomaly detector for intrusion detection usecase. Loda is trained to identify attacks in the form of bots from Netflow data. We use `cic_ids2017` benchmark dataset for the testing the performance of the model.

### Model Architecture

Loda (light weight online detector of anomalies), an ensemble of 1-D fixed histograms, where each histograms are built using random projection of features. The model is unsupervised anomaly detector where detection is done using negative log likelihood score.

### Requirements

Requirements can be installed with 
```
pip install -r requirements.txt
```

### Training

#### Training data
The dataset for the example used are from Canadian Institute for Cybersecurity (CIC). The CICIDS2017 (https://www.unb.ca/cic/datasets/ids-2017.html) dataset contains benign and the most up-to-date common attacks, which resembles the true real-world data (PCAPs). It also includes the results of the network traffic analysis using CICFlowMeter with labeled flows based on the time stamp, source, and destination IPs, source and destination ports, protocols and attack (CSV files). Also available is the extracted features definition. 


#### Training parameters

There are two main parameters used: number of random cuts for Loda and variance of the PCA transformation.
```
number_random_cuts = 1000
variance = 0.99
```
#### GPU Model
Tesla V100-SXM2

#### Model accuracy
The label distribution in the dataset is imbalanced, Average precision of 1.0 and Area under ROC curve of 0.74 is produced using test activity data.


#### Training script
To train the model, you can run the code in the notebook or alternatively, run the script under the `training-tunining-inference` directory using 
`$DATASET` path to extracted CIC dataset.
```bash
python training.py --input-name $DATASET/Monday-WorkingHours.pcap_ISCX.csv --model-name ../model/loda_ids
```

This will save trained model and config file under `model` directory.

### Inference
To run inference from trained model, load the trained Loda model and config parameters as follows:
```bash
python inference.py --input-name $DATASET/Friday-WorkingHours-Morning.pcap_ISCX.csv --config-path ../model/config.json --model-name ../model/loda_ids.npz
```
### How To Use This Model
This model is an example of intrusion detection model using unsupervised anomaly detector. This model requires  an aggregated netflow activity in the form of `cic_ids2017` format. Subset of the features used for training are described under `model/config.json`

### Input
The input is a netflow activity data collected in the form of tabular format.

### Output
The Unsupervised anomaly detector produce negative log likelihood as anomaly score of each data points. Large score indicates the more anomaly of the data point.  

#### Out-of-scope use cases
N/A

### Ethical considerations
N/A

### Reference
1. Sharafaldin, I.,Lashkari, A. H., & Ghorbani, A. A. (2018, January). Toward generating a new intrusion detection dataset and intrusion traffic characterization
2. Pevny,T. (2016). Loda: Lightweight on-line detector of anomalies. Machine Learning
