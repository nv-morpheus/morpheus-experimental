<!--
SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


# Cyber Foundation

# Model Overview

## Description:
* This model is a GPT model trained to generate synthetic Azure logs. This approach can be used to generate logs that are realistic for some downstream tasks, i.e. generating training data as a baseline, generating attack behavior to test detectors.  <br>

## Requirements:

* To run this example, additional requirements must be installed into your environment. A supplementary requirements file has been provided in this example directory.

`pip install -r requirements.txt`

## References(s): <br>

* https://github.com/karpathy/nanoGPT <br> 

## Model Architecture: <br>

**Architecture Type:** <br>

* Transformer <br>

**Network Architecture:** <br>

* GPT <br>

## Input: (Enter "None" As Needed) <br>

**Input Format:** <br>

* JSON <br>

**Input Parameters:** <br>

* Azure AD Logs <br>

**Other Properties Related to Output:** <br>

* N/A <br>

## Output: (Enter "None" As Needed) <br>

**Output Format:** <br>

* Text file with synthetic logs <br>

**Output Parameters:** <br>

* N/A <br>

**Other Properties Related to Output:**

* N/A <br> 

## Software Integration:<br>

**Runtime(s):** <br>

* Morpheus  <br>

**Supported Hardware Platform(s):** <br>

* Ampere/Turing <br>

**Supported Operating System(s):** <br>

* Linux <br>

## Model Version(s): 

* v1  <br>

# Training & Evaluation: 

## Training Dataset:

**Link:** <br>

* https://github.com/nv-morpheus/Morpheus/blob/main/models/datasets/training-data/azure/azure-ad-logs-sample-training-data.json  <br>

**Properties (Quantity, Dataset Descriptions, Sensor(s)):** <br>

* 3239 Azure AD logs <br>

**Dataset License:** <br>

* [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0) <br>

## Evaluation Dataset: <br>

**Link:** <br>

* N/A <br>

**Properties (Quantity, Dataset Descriptions, Sensor(s)):**

* N/A <br>

**Dataset License:** <br>

* N/A <br>

## Inference: <br>

**Engine:** <br>

* N/A <br>

**Test Hardware:** <br>

* A100  <br>

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

* The model is primarily designed for testing purposes and serves as a small pre-trained model used to generate Azure AD logs. 

### Fill in the blank for the model technique.

* This model is intended for developers who want to build GPT based synthetic log generator

### Name who is intended to benefit from this model. 

* The intended beneficiaries of this model are developers who aim to generate synthetic Azure logs.

### Describe the model output. 

* This model output is synthetic Azure AD logs. 

### List the steps explaining how this model works.

* This model is an example of a GPT model. This model requires raw log messages as input for training and a prompt for inference. The model is trained as in the training notebook. During inference, the trained model is prompted with the first key of the log type and generates synthetic logs.

### Name the adversely impacted groups (protected classes) this has been tested to deliver comparable outcomes regardless of:

* Not Applicable

### List the technical limitations of the model. 

* This model is trained with synthetic logs for demonstration purposes. A separate training is needed for other logs. 

### What performance metrics were used to affirm the model's performance?

* Intact raw logs

### What are the potential known risks to users and stakeholders?

* N/A

### What training is recommended for developers working with this model?

* None

### Link the relevant end user license agreement 

* [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0)<br>


## Model Card ++ Saftey & Security Subcard

### Link the location of the training dataset's repository.

* https://github.com/nv-morpheus/Morpheus/blob/main/models/datasets/training-data/azure/azure-ad-logs-sample-training-data.json

### Is the model used in an application with physical safety impact?

* No

### Describe physical safety impact (if present).

* N/A

### Was model and dataset assessed for vulnerability for potential form of attack?

* No

### Name applications for the model.

* This model is provided as an example of synthetic log generation. Users can create their own models for their use cases and downstream tasks.

### Name use case restrictions for the model.

* It's been trained with a small dataset for mainly demonstration purposes.  

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

* The dataset is initially reviewed upon addition, and subsequent reviews are conducted as needed or upon request for any changes.

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
