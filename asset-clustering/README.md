## Asset Clusering using Windows Event Logs

## Use Case
Cluster assets into various groups based on Windows Event Logs data.

### Version
1.0

### Model Overview
This model is a clustering algorithm to assign each host present in the dataset to a cluster based on aggregated and derived features from Windows Event Logs of that particular host.

### Model Architecture
The clustering algorithm is DBSCAN which stands for Density-Based Spatial Clustering of Applications with Noise. Input features to the model are derived from the windows event logs wherein various facets of login events, log off events etc.., are aggregated.

### Requirements
To run this example, additional requirements must be installed into your environment. A supplementary requirements file has been provided in this example directory.

```bash
pip install -r requirements.txt
```

### Training

#### Training data
MORE DETAILS AND CITATION HERE.

Trainign data consists of day-01, day-02, day-3 data from the lanl dataset. These 3 days is prep-processed and the features are aggregated. The resulting dataset contains 11400 hosts.


#### Training parameters
The following parameters are chosen in training for the DBSCAN algorithm:
- $\epsilon=0.0005$
- *Manhattan distance* as the metric i.e. Minkowski distance with $p=1$.


#### GPU model
V100

#### Model accuracy
clusters found = 9
Silhouette score = 0.975

#### Training script

To train the model run the following script under working directory.
```bash
cd ${MORPHEUS_EXPERIMENTAL_ROOT}/asset-clustering/training-tuning
# Run training script and save models
python model.py
```
CHANGE HERE from model.py to train.py. And save the trained model.

This saves trained model files under `../models` directory. Then the inference script can load the models for future inferences.

### How To Use This Model

### Input


### Output

#### Out-of-scope use cases
N/A

### Ethical considerations
N/A

### References
