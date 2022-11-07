## Detection of Malicious authentication  using Relational Graph Neural Network (RGCN)

## Use Case
Detection of Malicious Accounts on Azure-AD signon using Relational Graph Neural Network (RGCN)

### Version
1.0

### Model Overview

This model shows an application of a graph neural network for malicious authentication detection in Azure-AD signon heterogeneous graph. An Azure-AD signon dataset  includes four types of nodes, authentication, user, device and service application nodes are used for modeling. A Relational graph neural network (RGCN)  is used to identify malicious authentications.

### Model Architecture
It uses a  heterogeneous graph representation as input for RGCN. Since the input graph is heterogenous, an embedding for target node "authentication" is used for training the classifier.

### Requirements 
To run this example, additional requirements must be installed into your environment. A supplementary requirements file has been provided in this example directory.

```bash 
pip install -r requirements.txt
```

### Training
#### Training data

A training data consists of 45K authentication event, with semisupervised settings. The dataset is extracted from Azure-AD sign on events. The RGCN is trained to output embedded representation of authentication out of the graph and binary classification of individual authentication. 
#### Training epochs
30

#### Training batch size
1000

#### GPU model
V100

#### Model accuracy
AUC = 0.78 (RGCN binary classifier)


### How To Use This Model
This model is an example of a authentication detection pipeline using a graph neural network. This can be further retrained or fine-tuned to be used for similar types of  networks with similar graph structures.

### Input
Authentication data with nodes including user, authentication, device and service.

### Output
An anomalous score of authentication indicates a probability score of being malicious

#### Out-of-scope use cases
N/A

### Ethical considerations
N/A

### References
