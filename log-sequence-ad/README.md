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
The model is a sequence binary classifier trained with a vector representation of the log sequence of the BGL dataset. The task is to identify abnormal log sequences of alerts from a sequence of normally generated logs. This work is based on the model developed in the works of [[2](https://ieeexplore.ieee.org/document/9671642),[3](https://github.com/hanxiao0607/InterpretableSAD)], for further detail refer the paper and associated code at the reference link. <br>

## References(s):
1. https://arxiv.org/pdf/2202.04301.pdf
2. https://ieeexplore.ieee.org/document/9671642
3. https://github.com/hanxiao0607/InterpretableSAD<br> 

## Model Architecture:
**Architecture Type:** 
* LSTM binary classifier with word2vector embedding. <br>

**Network Architecture:** 
* LSTM and Word2Vec <br>

## Input
The input is an output of parsed system log messages represented as CSV file<br>
**Input Parameters:** 
```
output_dim = 2
emb_dim = 8
hidden_dim = 128
n_layers = 1
dropout = 0.0
batch_size = 32
n_epoch = 10
```
<br>

**Input Format:** 
* CSV<br>

**Other Properties Related to Output:** 
* None<br>

## Output
Binary classifier output is assigned to each sequence log message in the input file. The predicted output is appended to the last column of the input sequence.<br>

**Output Parameters:**  
* None <br>

**Output Format:** 
* CSV (log rows)<br>

**Other Properties Related to Output:** 
* None <br> 
## Software Integration:
**Runtime(s):** 
* Pytorch  <br>

**Supported Hardware Platform(s):** <br>
* Ampere/Turing <br>

**Supported Operating System(s):** <br>
* Linux <br>
  
## Model Version(s): 
1.0 <br>

# Training & Evaluation: 

## Training Dataset:

**Link:** 
* [BlueGeme/L](https://zenodo.org/record/3227177/files/BGL.tar.gz?download=1) <br>

**Properties (Quantity, Dataset Descriptions, Sensor(s)):** 
* The dataset for the example used is from BlueGene/L Supercomputer System (BGL). BGL dataset contains 4,747,963 log messages from supercomputer system at Lawrence Livermore National Labs. The model is trained and evaluated using 1 million rows of preprocessed logs using [Drain](https://github.com/logpai/logparser) parser<br>

**Dataset License:**  
* [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0) <br>

## Evaluation Dataset:
**Link:**  
* [BGL-evaluation sample](https://github.com/LogIntelligence/LogPPT/blob/master/logs/BGL/BGL_2k.log_structured.csv)  <br>

**Properties (Quantity, Dataset Descriptions, Sensor(s)):** 
* Processed 39K BGL log dataset. <br>

**Dataset License:** 
* [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0) <br>

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
* The model is primarily designed for testing purposes and serves as a small pretrained model specifically used to evaluate and validate the log sequence anomaly detection usecase.
### Fill in the blank for the model technique.
* This model is intended for developers that want to build log-sequence based anomaly detection.

### Name who is intended to benefit from this model. 
* The intended beneficiaries of this model are developers who aim to test the performance and functionality of the log sequence detector using public log datasets.
 It may not be suitable or provide significant value for real-world logs analysis.
### Describe the model output.
* This model outputs binary prediction of being anomaly or not.

### List the steps explaining how this model works.
* This model is an example of a sequence binary classifier. This model requires parsed log messages as input for training and inference. The model and Word2Vector embedding is trained as follows in the training notebook. During inference, the trained model is loaded from `model` directory, and input files in the form of parsed logs are expected to output prediction for sequences of log messages.<br>

### Name the adversely impacted groups (protected classes) this has been tested to deliver comparable outcomes regardless of:
* Not Applicable

### List the technical limitations of the model.
* The model expects system logs with specific features that match the training dataset. This model requires parsed log messages as input for training and inference.

### What performance metrics were used to affirm the model's performance?
* The model is evaluated using F1 score, accuracy for the ability to identify abnormal log sequence from set of sequence logs. 

### What are the potential known risks to users and stakeholders? 
* None
### What training is recommended for developers working with this model?  If none, please state "none."
* None
### Link the relevant end user license agreement 
* [Apache 2.0](https://github.com/nv-morpheus/Morpheus/blob/branch-25.06/LICENSE)

## Model Card ++ Saftey & Security Subcard

### Link the location of the training dataset's repository (if able to share).
* [BlueGeme/L](https://zenodo.org/record/3227177/files/BGL.tar.gz?download=1)

### Is the model used in an application with physical safety impact?
* No

### Describe physical safety impact (if present).
* Not Applicable

### Was model and dataset assessed for vulnerability for potential form of attack?
* No
### Name applications for the model.
* Typically used for testing to identify abnormality out of sequence of logs
### Name use case restrictions for the model.
* Only tested for log sequence using the described parsed logs, it may not be suitable for other applications.
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
* The data used in this model is obtained from public shared data [BlueGeme/L](https://zenodo.org/record/3227177/files/BGL.tar.gz?download=1). There are no privacy concerns or PII involved in this data.


### Protected classes used to create this model? (The following were used in model the model's training:)

* Not applicable

### How often is dataset reviewed?
* Not applicable. The dataset is fully hosted and maintained by external source of [Zenodo](https://zenodo.org). Users can refer the main site dataset.

### Is a mechanism in place to honor data
* Yes
### If PII collected for the development of this AI model, was it minimized to only what was required? 
* Not applicable

### Is data in dataset traceable?
* No

### Scanned for malware?
* No
### Are we able to identify and trace source of dataset?
* Yes
### Does data labeling (annotation, metadata) comply with privacy laws?
* Not applicable
### Is data compliant with data subject requests for data correction or removal, if such a request was made?
* Not applicable