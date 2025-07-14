<!--
SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Model Overview

### Description:
The model uses a graph clustering approach (cited below) which assigns each host present in the dataset to a cluster based on 
1. Aggregated and derived features from sflow Logs of that particular host
2. The host connectivity to adjacent assets in the graphical representation (derived from sflow logs)

## References(s):
[1]. H. Zhang, P. Li, R. Zhang and X. Li, "Embedding Graph Auto-Encoder for Graph Clustering," in IEEE Transactions on Neural Networks and Learning Systems, doi: 10.1109/TNNLS.2022.3158654.

## Model Architecture:
The model architecture was proposed in the EGAE paper below (cited). Inputs of EGAE consist of two parts, graph and features. After encoding, data are mapped into a latent feature space as part of the encoder module. There are two decoder modules: 
1. Decoder for clustering: Relaxed k-means is embedded into GAE to induce it to generate preferable embeddings. 
2. Decoder for Graph : Optimize (minimize) reconstruction error

**Architecture Type:** 
* Graph Neural Network <br>

**Network Architecture:** 
* Graph Autoencoder with 2-layers<br>

## Input
* The input is  Sflow data from ~3000 devices
Armis device and application data<br>

**Input Parameters:**
* None
<br>

**Input Format:** 
* CSV format<br>

**Other Properties Related to Output:** 
* None<br>
## Output
* Clustering information and cluster membership<br>

**Output Parameters:**  
* None <br>

**Output Format:** 
* CSV<br>

## Software Integration:
**Runtime(s):** 
* cupy <br>

**Supported Hardware Platform(s):** <br>
* Ampere/Turing <br>

**Supported Operating System(s):** <br>
* Linux <br>
  
## Model Version(s): 
1.0 <br>

# Training & Evaluation: 

## Training Dataset:
 
**Link:** 
* [Sflow](dataset/)<br>

**Properties (Quantity, Dataset Descriptions, Sensor(s)):** 
* The dataset uses Sflow data to come up with a graph representation where each node in the graph is an asset. Since sflow data is directional, we use 'source' as the target asset. The feature matrix for this asset is created using derived and aggregated features from sflow data and armis data. The adjacency matrix is derived using the graph representation of the devices from sflow data. Each row in the resulting dataset is an asset and can be uniquely identified by the mac address. All information in the Sflow is obfuscated to remove any private information<br>

**Dataset License:** 
* [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0)<br>
## Evaluation Dataset:
**Link:** 
* [Sflow](dataset/)<br>  <br>

**Properties (Quantity, Dataset Descriptions, Sensor(s)):** 
* Subset of the simulated and obfuscated Sflow <br>

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
* English (100%)

### What is the geographic origin language balance of the model validation data?
* Not Applicable

### What is the educational background balance of the model validation data?
* Not Applicable

### What is the accent balance of the model validation data?
* Not Applicable
### Describe measures taken to mitigate against unwanted bias.
* Not Applicable

## Model Card ++ Explainability Subcard

### Name example applications and use cases for this model. 
* The model is primarily designed for testing purposes and serves as a small pretrained model specifically used to evaluate and validate  asset clustering application using GNN.

### Fill in the blank for the model technique.
* This model is intended for developers that want to build asset clustering application using GNN.

### Name who is intended to benefit from this model. 
* The intended beneficiaries of this model are developers who aim to test the performance and functionality of the asset clustering application pipeline using sflow datasets.

### Describe the model output.
* This model outputs cluster membership of devices based on sflow activities.
### List the steps explaining how this model works. (e.g., )  
* The model architecture was proposed in the EGAE paper [1]. Inputs of EGAE consist of two parts, graph and features. After encoding, data are mapped into a latent feature space as part of the encoder module. There are two decoder modules. 
    - Decoder for clustering: Relaxed k-means is embedded into GAE to induce it to generate preferable embeddings.
    - Decoder for Graph : Optimize (minimize) reconstruction error
<br>

### Name the adversely impacted groups (protected classes) this has been tested to deliver comparable outcomes regardless of:
* Not Applicable

### List the technical limitations of the model.
* This model requires feature engineered Sflow activity data along ARMIS device enrichment.

### What performance metrics were used to affirm the model's performance?
* Silhouette plot and score

### What are the potential known risks to users and stakeholders? 
* Not Applicable

### What training is recommended for developers working with this model?  If none, please state "none."
* None 

### Link the relevant end user license agreement 
* [Apache 2.0](https://github.com/nv-morpheus/Morpheus/blob/branch-25.10/LICENSE)

## Model Card ++ Saftey & Security Subcard

### Link the location of the training dataset's repository (if able to share).
* [Dataset](./dataset)

### Is the model used in an application with physical safety impact?
* No

### Describe physical safety impact (if present).
* None

### Was model and dataset assessed for vulnerability for potential form of attack?
* No
### Name applications for the model.
* Typically used to cluster ARMIS devices in network based on Sflow activities.
### Name use case restrictions for the model.
* The model is trained in the format of Sflow dataset schema, the model might not be suitable for other applications.
### Has this been verified to have met prescribed quality standards?
* No

### Name target quality Key Performance Indicators (KPIs) for which this has been tested.  
* Not Applicable
### Technical robustness and model security validated?
* Not Applicable
### Is the model and dataset compliant with National Classification Management Society (NCMS)?
* Not Applicable

###  Are there explicit model and dataset restrictions?
* No

### Are there access restrictions to systems, model, and data?
* No
### Is there a digital signature?
* No

## Model Card ++ Privacy Subcard


### Generatable or reverse engineerable personally-identifiable information (PII)?

* Neither

### Was consent obtained for any PII used?
* Not Applicable, The synthetic data used in this model is generated using the [faker](https://github.com/joke2k/faker/blob/master/LICENSE.txt)  python package. The device information field is generated by faker, which pulls items from its own dataset of fictitious values (located in the linked repo). There are no privacy concerns or PII involved in this synthetic data generation process.

### Protected classes used to create this model? (The following were used in model the model's training:)

* Not applicable


### How often is dataset reviewed?
* The dataset is initially reviewed upon addition, and subsequent reviews are conducted as needed or upon request for any changes.

### Is a mechanism in place to honor data
* No (as the dataset is fully synthetic)
### If PII collected for the development of this AI model, was it minimized to only what was required? 
* Not Applicable (no PII collected)

### Is data in dataset traceable?
* No

### Scanned for malware?
* No
### Are we able to identify and trace source of dataset?
* Yes at ([Dataset](./dataset))
### Does data labeling (annotation, metadata) comply with privacy laws?
* Not applicable
### Is data compliant with data subject requests for data correction or removal, if such a request was made?
* Not applicable