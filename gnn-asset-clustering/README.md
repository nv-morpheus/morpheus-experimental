## Asset Clustering using Graph based Approach

## Use Case
Cluster assets into various groups based on SFlow and Armis enriched data.

### Version
1.0

### Model Overview
The model uses a graph clustering approach (cited below) which assigns each host present in the dataset to a cluster based on 
1. Aggregated and derived features from sflow Logs of that particular host
2. The host connectivity to adjacent assets in the graphical representation (derived from sflow logs)

### Model Architecture
The model architecture was proposed in the EGAE paper below (cited). Inputs of EGAE consist of two parts, graph and features. After encoding, data are mapped into a latent feature space as part of the encoder module. There are two decoder modules: 
1. Decoder for clustering: Relaxed k-means is embedded into GAE to induce it to generate preferable embeddings. 
2. Decoder for Graph : Optimize (minimize) reconstruction error

### Requirements
TBD


### Training

#### Training data
Internal NVIDIA Sflow data from ~3000 devices
Armis device and application data

We use sflow data to come up with a graph representation where each node in the graph is an asset. Since sflow data is directional, we use 'source' as the target asset. The feature matrix for this asset is created using derived and aggregated features from sflow data and armis data. The adjacency matrix is derived using the graph representation of the devices from sflow data. Each row in the resulting dataset is an asset and can be uniquely identified by the mac address.

### Ethical considerations
N/A

### References
[1]. H. Zhang, P. Li, R. Zhang and X. Li, "Embedding Graph Auto-Encoder for Graph Clustering," in IEEE Transactions on Neural Networks and Learning Systems, doi: 10.1109/TNNLS.2022.3158654.
