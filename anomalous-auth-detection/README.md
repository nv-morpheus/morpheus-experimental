<!--
SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

## Model Overview

### Description:
Azure active directory (Azure-AD) is an identity and access management service, that helps users to access external and internal resources such as Office365, SaaS applications. The Sign-in logs in Azure-AD log identifies who the user is, how the application is used for the access, and the target accessed by the identity [1](https://docs.microsoft.com/en-us/azure/active-directory/reports-monitoring/concept-sign-ins). On a given time 𝑡, a service 𝑠 is requested by user 𝑢 from device 𝑑 using the authentication mechanism of 𝑎 to be either allowed or blocked. For detailed explanation refer to the following page of the [Microsoft azure-sign-in](https://docs.microsoft.com/en-us/azure/active-directory/reports-monitoring/concept-sign-ins)

Similar works on anomalous authentication detection include applying blackbox ML models on handcrafted features extracted from authentication logs or rule-based models. This workflow closely follows the success of heterogenous GNN embedding on cyber applications such as fraud detection [[2](https://doi.org/10.1145/3269206.3272010),[5](https://www.vldb.org/pvldb/vol15/p427-rao.pdf)], cyber-attack detection on prevalence dataset [[3](http://arxiv.org/abs/2112.08986)]. Unlike earlier models, this work uses a heterogenous graph for authentication graph modeling and relational GNN embedding for capturing relations among different entities. This allows us to take advantage of relations among users/services, and at the same time avoids the feature-extracting phase. In the end, the model learns both from the structural identity and unique feature identity of individual users. 
<!--
The main motivation behind this work is to tackle the drawbacks of rule-based or feature-based systems, such as failure to generalize for new attacks and the requirements for dynamic rules that need to be maintained often. An evolving attack and connected anomalous users across the network are hard to detect through feature/rule-based methods. Additionally, using the graph approach minimizes efforts on feature engineering tasks.
-->
 <br>

## References(s):
1. https://docs.microsoft.com/en-us/azure/active-directory/reports-monitoring/concept-sign-ins
2. Liu, Ziqi, et al. “Heterogeneous Graph Neural Networks for Malicious Account Detection.” arXiv [cs.LG], 27 Feb. 2020, https://doi.org/10.1145/3269206.3272010. arXiv.
3. Lv, Mingqi, et al. “A Heterogeneous Graph Learning Model for Cyber-Attack Detection.” arXiv [cs.CR], 16 Dec. 2021, http://arxiv.org/abs/2112.08986. arXiv.
4. Schlichtkrull, Michael, et al. "Modeling relational data with graph convolutional networks." European semantic web conference. Springer, Cham, 2018 https://arxiv.org/abs/1703.06103
5. Rao, Susie Xi, et al. "xFraud: explainable fraud transaction detection." Proceedings of the VLDB Endowment 3 (2021) https://www.vldb.org/pvldb/vol15/p427-rao.pdf
6. Powell, Brian A. "Detecting malicious logins as graph anomalies." Journal of Information Security and Applications 54 (2020): 102557 <br> 

## Model Architecture: 
It uses a  heterogeneous graph representation as input for RGCN. Since the input graph is heterogenous, an embedding for target node "authentication" is used for training the RGCN classifier. The model is trained as a binary classifier with the task to output "success" or "failure" to each authentication embedding.<br>
**Architecture Type:** 
* Graph Neural Network <br>

**Network Architecture:** 
* 2-layer RGCN with 8 dimension output embedding <br>

## Input
Authentication data with nodes including user, authentication, device, and service.<br>
**Input Parameters:** 
* None <br>

**Input Format:** 
* JSON format<br>

**Other Properties Related to Output:** 
* None<br>

## Output
* An anomalous score of authentication indicates a probability score of being an anomaly. A threshold of e.g 0.49 could be used to output produce "benign"
or "fraudulent" authentication.<br>

**Output Parameters:** 
* None <br>

**Output Format:** 
* CSV (scores & authenticationId) <br>

**Other Properties Related to Output:** 
* None <br> 
## Software Integration:
**Runtime(s):** 
* Pytorch
* DGL  <br>


**Supported Hardware Platform(s):** <br>
* Ampere/Turing <br>

**Supported Operating System(s):** <br>
* Linux <br>
  
## Model Version(s): 
1.0 <br>

# Training & Evaluation: 

## Training Dataset:
**Link:**  
* [training data subset](dataset/azure_synthetic/azure_ad_logs_sample_with_anomaly_all.json) <br>

**Properties (Quantity, Dataset Descriptions, Sensor(s)):**
*  A training data consists of 1992 authentication event, with a label indicating either failure or success. The dataset is simulated to resemble Azure-AD sign on events. <br>

**Dataset License:**  
*  [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0)<br>
## Evaluation Dataset:
**Link:** 
* [evaluation data subset](dataset/azure_synthetic/azure_ad_logs_sample_with_anomaly_all.json) <br>

**Properties (Quantity, Dataset Descriptions, Sensor(s)):** 
* A evaluation data consists of 235 authentication event, with a label indicating either failure or success. <br>

**Dataset License:** 
* [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0)<br>

## Inference:
**Engine:** 
* Pytorch <br>

**Test Hardware:** <br>
* Other (Not Listed)  <br>

# Subcards
## Model Card ++ Bias Subcard

### What is the gender balance of the model validation data?
* Not Applicable

### What is the racial/ethnicity balance of the model validation data?
* Not Applicable

### What is the age balance of the model validation data?
* Not Applicable

### What is the language balance of the model validation data?
* English: 100%

### What is the geographic origin language balance of the model validation data?
* Not Applicable

### What is the educational background balance of the model validation data?
* Not Applicable

### What is the accent balance of the model validation data?
* Not Applicable

### What is the face/key point balance of the model validation data?
* Not Applicable

### What is the skin/tone balance of the model validation data?
* Not Applicable

### What is the religion balance of the model validation data?
* Not Applicable

### Individuals from the following adversely impacted (protected classes) groups participate in model design and testing.
* Not Applicable

### Describe measures taken to mitigate against unwanted bias.
* Not Applicable

## Model Card ++ Explainability Subcard

### Name example applications and use cases for this model. 
* The model is primarily designed for testing purposes and serves as a small pretrained model specifically used to evaluate and validate the RGCN model. Its application is focused on assessing the effectiveness of the pipeline rather than being intended for broader use cases or specific applications beyond testing.

### Fill in the blank for the model technique.
* This model is intended for developers that want to build and/or customize Relational graph neural network (RGCN) for authentication detection.

### Name who is intended to benefit from this model. 

* The intended beneficiaries of this model are developers who aim to test the performance and functionality of the RGCN pipeline using synthetic datasets. It may not be suitable or provide significant value for real-world Azure-log analysis.

### Describe the model output.
* This model outputs an anomalous score of authentication indicates a probability score of being an anomaly. A threshold of e.g 0.49 could be used to output produce "benign" or "fraudulent" authentication.

### List the steps explaining how this model works. (e.g., )  
* An Azure-AD sign-in dataset it includes four types of nodes, authentication, user, device and service application nodes are used for modeling. This model shows an application of a graph neural network for anomalous authentication detection in Azure-AD sign-in using heterogeneous graph. A Relational graph neural network (RGCN)  is used to identify anomalous authentications.
### Name the adversely impacted groups (protected classes) this has been tested to deliver comparable outcomes regardless of:
* Not Applicable

### List the technical limitations of the model. 
* This model version is trained on a simulated Azure-AD sign-on logs schema, with entities (user, service, device, authentication) and "statsFlag" as requirements. Data lacking the required features or requiring a different feature set may not be compatible with the model.

### What performance metrics were used to affirm the model's performance? 
* The model is evaluated using Area under ROC curve and accuracy for authentications.

### What are the potential known risks to users and stakeholders? 
* None

### What training is recommended for developers working with this model?  If none, please state "none."
* None
### Link the relevant end user license agreement 
* [Apache 2.0](https://github.com/nv-morpheus/Morpheus/blob/branch-23.07/LICENSE)

## Model Card ++ Saftey & Security Subcard

### Link the location of the training dataset's repository (if able to share).
* [training dataset](dataset/azure_synthetic/azure_ad_logs_sample_with_anomaly_all.json)

### Is the model used in an application with physical safety impact?
* No

### Describe physical safety impact (if present).
* Not Applicable


### Was model and dataset assessed for vulnerability for potential form of attack?
* Not Applicable (synthetically generated)
### Name applications for the model.
* Anomalous azure authentication detection
### Name use case restrictions for the model.
* This model version requires Azure-AD sign-on logs schema, with entities (user, service, device, authentication) and "statsFlag" as requirements, the primary application for this model is for testing the pipeline.
### Has this been verified to have met prescribed quality standards?
* No

### Name target quality Key Performance Indicators (KPIs) for which this has been tested.
* None

### Technical robustness and model security validated?
* No

### Is the model and dataset compliant with National Classification Management Society (NCMS)?
* No

### Are there explicit model and dataset restrictions?
* No

### Are there access restrictions to systems, model, and data?
* No

### Is there a digital signature?
* No


## Model Card ++ Privacy Subcard


### Generatable or reverse engineerable personally-identifiable information (PII)?
* Neither

### Was consent obtained for any PII used?
* The synthetic data used in this model is generated using the [faker](https://github.com/joke2k/faker/blob/master/LICENSE.txt)  python package. The user agent field is generated by faker, which pulls items from its own dataset of fictitious values (located in the linked repo). Similarly, the event source field is randomly chosen from a list of event names provided in the AWS documentation. There are no privacy concerns or PII involved in this synthetic data generation process.

### Protected classes used to create this model? (The following were used in model the model's training:)
* Not applicable

### How often is dataset reviewed?
* The dataset is initially reviewed upon addition, and subsequent reviews are conducted as needed or upon request for any changes.

### Is a mechanism in place to honor data subject right of access or deletion of personal data?
* No (as the dataset is fully synthetic)

### If PII collected for the development of this AI model, was it minimized to only what was required?
* Not Applicable (no PII collected)

### Is data in dataset traceable?
* No

### Scanned for malware?
* No

### Is data in dataset traceable?
* No

### Scanned for malware?
* No

### Are we able to identify and trace source of dataset?
* Yes, [training dataset](dataset/azure_synthetic/azure_ad_logs_sample_with_anomaly_all.json)


### Does data labeling (annotation, metadata) comply with privacy laws?
* Not Applicable

### Is data compliant with data subject requests for data correction or removal, if such a request was made?
* Not Applicable (as data is synthetic)