## Detection of Malicious authentication  using Relational Graph Neural Network (RGCN)

## Use Case
Detection of Malicious Accounts on Azure-AD signon using Relational Graph Neural Network (RGCN)

### Version
1.0

### Model Overview

This model shows an application of a graph neural network for malicious authentication detection in Azure-AD signon heterogeneous graph. An Azure-AD signon dataset  includes four types of nodes, authentication, user, device and service application nodes are used for modeling. A Relational graph neural network (RGCN)  is used to identify malicious authentications.

### Model Architecture
It uses a  heterogeneous graph representation as input for RGCN. Since the input graph is heterogenous, an embedding for target node "authentication" is used for training the RGCN classifier. The model is trained as binary classifier with task to output "success" or "failure" to each authentication embedding.

### Requirements 
To run this example, additional requirements must be installed into your environment. A supplementary requirements file has been provided in this example directory.

```bash 
pip install -r requirements.txt
```

### Training
#### Training data

A training data consists of 1.5K authentication event, with label indicating either failure or success. The dataset is simulated to resemble Azure-AD sign on events. The RGCN is trained to output embedded representation of authentication out of the graph and binary classification of each authentication. 
#### Training epochs
30

#### Training batch size
1000

#### GPU model
V100

#### Model accuracy
AUC = 0.75 (RGCN binary classifier)
Accuracy = 0.85


### How To Use This Model
This model is an example of a authentication detection pipeline using a graph neural network. This model requires an authentication graph such as Azure with four nodes as main entities (user, service, device, authentication) and "statsFlag" for semi-supervision as its main core. The "authentication" node is the target node for training/prediction.

### Input
Authentication data with nodes including user, authentication, device and service.

### Output
An anomalous score of authentication indicates a probability score of being malicious. A threshold of e.g 0.49 could be used to output produce "benign"
or "fraudulent" authentication.

#### Out-of-scope use cases
This model version is trained on a simulated Azure-AD signon logs schema, with entities (user, service, device, authentication) and "statsFlag" as requirement.

### Ethical considerations
N/A

### References

1. https://docs.microsoft.com/en-us/azure/active-directory/reports-monitoring/concept-sign-ins
2. Liu, Ziqi, et al. “Heterogeneous Graph Neural Networks for Malicious Account Detection.” arXiv [cs.LG], 27 Feb. 2020, https://doi.org/10.1145/3269206.3272010. arXiv.
3. Lv, Mingqi, et al. “A Heterogeneous Graph Learning Model for Cyber-Attack Detection.” arXiv [cs.CR], 16 Dec. 2021, http://arxiv.org/abs/2112.08986. arXiv.
4. Schlichtkrull, Michael, et al. "Modeling relational data with graph convolutional networks." European semantic web conference. Springer, Cham, 2018
5. Rao, Susie Xi, et al. "xFraud: explainable fraud transaction detection." Proceedings of the VLDB Endowment 3 (2021)
6. Powell, Brian A. "Detecting malicious logins as graph anomalies." Journal of Information Security and Applications 54 (2020): 102557