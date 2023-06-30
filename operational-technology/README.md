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


# Industrial Control System (ICS) Cyber Attack Detection

# Model Overview

## Description:
* The model is a multi-class XGBoost classifier that predicts each event on a power system based on dataset features. <br>

## References(s):

* https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets <br>

* http://www.ece.uah.edu/~thm0009/icsdatasets/PowerSystem_Dataset_README.pdf <br>

* Morris, T., Thornton, Z., Turnipseed, I.,  Industrial Control System Simulation and Data Logging for Intrusion Detection System Research. 7th Annual Southeastern Cyber Security Summit. Huntsvile, AL. June 3 - 4, 2015.

* Beaver, Justin M., Borges-Hink, Raymond C., Buckner, Mark A., "An Evaluation of Machine Learning Methods to Detect Malicious SCADA Communications," in the Proceedings of 2013 12th International Conference on Machine Learning and Applications (ICMLA), vol.2, pp.54-59, 2013. doi: 10.1109/ICMLA.2013.105 http://www.google.com/url?q=http%3A%2F%2Fieeexplore.ieee.org%2Fstamp%2Fstamp.jsp%3Ftp%3D%26arnumber%3D6786081%26isnumber%3D6786067&sa=D&sntz=1&usg=AOvVaw2358QIW3gbS8nsykJ3DFgl

## Model Architecture: 

**Architecture Type:** 

* Gradient Boosting <br>

**Network Architecture:**

* XGBOOST <br>

## Input: (Enter "None" As Needed)

**Input Format:** 

* Tabular format in which the dataset features contain synchrophasor measurements and data logs from Snort, a simulated control panel, and relays.  <br>

**Input Parameters:**

* N/A <br>

**Other Properties Related to Output:**

* N/A <br>

## Output: (Enter "None" As Needed)

**Output Format:** 

* Natural Events, No Events or Attack Events <br>

**Output Parameters:**

* N/A  <br>

**Other Properties Related to Output:**

* N/A <br> 

## Requirements

* Requirements can be installed with

`pip install -r requirements.txt`
* and for `p7zip`

`apt update`
`apt install p7zip-full p7zip-rar`

## Software Integration:

**Runtime(s):** 

* Morpheus  <br>

**Supported Hardware Platform(s):** <br>

* Ampere/Turing <br>

**Supported Operating System(s):** <br>

* Linux <br>

## Model Version(s): 

* v1  <br>

# Training & Evaluation: 

## Training Dataset:

**Link:** 

* http://www.ece.uah.edu/~thm0009/icsdatasets/triple.7z <br>

**Properties (Quantity, Dataset Descriptions, Sensor(s)):**

* There are 78377 rows in the dataset. <br>

**Dataset License:**

* https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets <br>

## Evaluation Dataset:
**Link:** 

* http://www.ece.uah.edu/~thm0009/icsdatasets/triple.7z <br>

**Properties (Quantity, Dataset Descriptions, Sensor(s)):**

* There are 78377 rows in the dataset. <br>

**Dataset License:**

* https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets <br>

## Inference:

**Engine:** 

* Triton <br>

**Test Hardware:** <br>

* Other  <br>

# Subcards

## Model Card ++ Bias Subcard

### What is the gender balance of the model validation data?  

* Not Applicable

### What is the racial/ethnicity balance of the model validation data?

* Not Applicable

### What is the age balance of the model validation data?

* Not Applicable

### What is the language balance of the model validation data?

* Not Applicable

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

*  It's primarily for testing purposes Natural Events, No Events and Attack Events can be detected in an Industrial Control System

### Fill in the blank for the model technique.

* This model is intended for developers who want to test the model that can detect different events in the example Operational Technologies dataset.

### Name who is intended to benefit from this model. 

* This model is intended for users who want to test with models that can differentiate Natural Events, No Events and Attack Events.

### Describe the model output. 

* This model outputs one of these results: Natural Events, No Events and Attack Events

### List the steps explaining how this model works. 

* An XGBoost model gets trained with the dataset, and in inference, the model predicts one of the multiple classes for each row.

### Name the adversely impacted groups (protected classes) this has been tested to deliver comparable outcomes regardless of:
* Not Applicable

### List the technical limitations of the model. 

* Further training is needed for different data types.

### What performance metrics were used to affirm the model's performance?

* F1

### What are the potential known risks to users and stakeholders?

* N/A

### What training is recommended for developers working with this model?

* None

### Link the relevant end user license agreement 

* [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0)<br>


## Model Card ++ Saftey & Security Subcard

### Link the location of the training dataset's repository.

* http://www.ece.uah.edu/~thm0009/icsdatasets/triple.7z

### Is the model used in an application with physical safety impact?

* No

### Describe physical safety impact (if present).

* N/A

### Was model and dataset assessed for vulnerability for potential form of attack?

* No

### Name applications for the model.

*  Industrial Control System (ICS) Cyber Attack Detection

### Name use case restrictions for the model.

* Different models need to be trained for different types of data

### Has this been verified to have met prescribed quality standards?

* No

### Name target quality Key Performance Indicators (KPIs) for which this has been tested.  

* N/A

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
* N/A

### Protected classes used to create this model? (The following were used in model the model's training:)
* N/A

### How often is dataset reviewed?
* Unknown

### Is a mechanism in place to honor data subject right of access or deletion of personal data?

* N/A

### If PII collected for the development of this AI model, was it minimized to only what was required? 
* N/A

### Is data in dataset traceable?
* N/A

### Scanned for malware?
* No

### Are we able to identify and trace source of dataset?
* Yes

### Does data labeling (annotation, metadata) comply with privacy laws?
* N/A

### Is data compliant with data subject requests for data correction or removal, if such a request was made?
* N/A
