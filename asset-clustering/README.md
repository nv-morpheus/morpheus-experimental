## Asset Clusering using Windows Event Logs

## Use Case
Cluster assets into various groups based on Windows Event Logs data.

### Version
1.0

### Model Overview
The model is a clustering algorithm to assign each host present in the dataset to a cluster based on aggregated and derived features from Windows Event Logs of that particular host.

### Model Architecture
There are two clustering algorithms available: 
- DBSCAN which stands for Density-Based Spatial Clustering of Applications with Noise.
- KMeans
Input features to the model are derived from the windows event logs wherein various facets of login events, type of logon event, number of usernames associated with a host etc.., are aggregated.

### Requirements
An environemnt based on __[Rapids](https://rapids.ai/pip.html)__ is required to run the scripts and python notebook provided. Also on top of that th additional requirements can be installed into the environment via the supplementary requirements file provided.

```bash
pip install -r requirements.txt
```

### Training

#### Training data
In this project we use the publicly available __[**Unified Host and Network Data Set**](https://csr.lanl.gov/data/2017/)__[1] dataset from the Advanced Research team in Cyber Systems of the Los Alamos National Laboratory to demonstrate various aspects involved in clustering assets in a given network.
The lanl dataset consists of netflow and windows event log (wls) files for 90 days. For this project we focus solely on the windows event log files which use the naming convention wls_day-01.bz2, wls_day-02.bz2,..., wls_day-90.bz2. The training data uses first ten days of data i.e. wls_day-01.bz2,..., wls_day-10.bz2. These ten days' data is pre-processed and the features are aggregated. The resulting dataset contains 14044 hosts and is saved in datasets/host_agg_data_day-01_day-10.csv.


#### Training parameters
The following parameters are chosen in training for the DBSCAN algorithm:
- $\epsilon=0.0005$
- *Manhattan distance* as the metric i.e. Minkowski distance with $p=1$.


#### Model accuracy
clusters found = 9 (+1 cluster for for the noisy samples)
Silhouette score = 0.975

#### Training script

To train the model run the following script under working directory.
```bash
cd ${MORPHEUS_EXPERIMENTAL_ROOT}/asset-clustering/training-tuning-inference
# Run training script and save models
python train.py --model dbscan
```
This saves trained model files under `../models` directory. Then the inference script can load the models for future inferences.

### Inference Input

```
python inferencepy --model dbscan
```
Whent the above comand is executed, dbscan clustering is performed on the windows event logs data from days 11 to 15. This data is pre-processed and aggregated to a validation dataset which can be found at datasets/host_agg_data_day-11_day-15.csv. This contains a total of 12606 hosts.


### Inference Output
The clustering of the 12606 hosts is performed and the count of each cluster is printed to stdout.

### Ethical considerations
N/A

### References
[1]. M. Turcotte, A. Kent and C. Hash, “Unified Host and Network Data Set”, in Data Science for Cyber-Security. November 2018, 1-22
