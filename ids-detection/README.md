# Intrusion Detection Using Lightweight Online Detector of Anomalies (LODA)

## Model Overview

### Description:
The model is  a Lightweight Online Detector of Anomalies (Loda) anomaly detector for intrusion detection use cases. Loda is trained to identify attacks in the form of bots from Netflow data. We used `cic_ids2017` benchmark dataset for testing the performance of the model.
## References(s):
1. Sharafaldin, I.,Lashkari, A. H., & Ghorbani, A. A. (2018, January). Toward generating a new intrusion detection dataset and intrusion traffic characterization
2. Pevny,T. (2016). Loda: Lightweight on-line detector of anomalies. Machine Learning<br> 

## Model Architecture:
Loda (lightweight online detector of anomalies), an ensemble of 1-D fixed histograms, where each histogram are built using random projection of features. The model is an unsupervised anomaly detector where detection is scored using a negative log-likelihood score.
**Architecture Type:** LODA <br>
**Network Architecture:** N/A<br>

## Input
The input is Netflow activity data collected in the form of a tabular format.<br>
**Input Parameters:**
```
number_random_cuts = 1000
variance = 0.99
```
<br>

## Output
The Unsupervised anomaly detector produces negative log-likelihood as the anomaly score of each data point. A large score indicates anomalousness of data points <br>
**Output Parameters:** None <br>
## Software Integration:
**Runtime(s):** cupy <br>

**Supported Hardware Platform(s):** <br>
* Ampere/Turing <br>

**Supported Operating System(s):** <br>
* Linux <br>
  
## Model Version(s): 
1.0 <br>

# Training & Evaluation: 

## Training Dataset:
 
**Link:** [CICIDS2017]('(https://www.unb.ca/cic/datasets/ids-2017.html)) <br>
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** The dataset is from Canadian Institute for Cybersecurity (CIC). The CICIDS2017  dataset contains benign and the most up-to-date common attacks, which resembles the true real-world data (PCAPs). It also includes the results of the network traffic analysis using CICFlowMeter with labeled flows based on the time stamp, source, and destination IPs, source and destination ports, protocols, and attack (CSV files). Also available is the extracted features definition.<br>
**Dataset License:** N/A  <br>

## Evaluation Dataset:
**Link:**  [CICIDS2017]('(https://www.unb.ca/cic/datasets/ids-2017.html))  <br>
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** Subset of CICIDS2017 with only botnet attacks. 
**Dataset License:** N/A<br>

## Inference:
**Engine:** cupy
**Test Hardware:** <br>
* Other (Not Listed)  <br>

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
### Describe measures taken to mitigate against unwanted bias.
* Not Applicable

## Model Card ++ Explainability Subcard

### Name example applications and use cases for this model. 
* This model is intended to be used for IDS application.
### Fill in the blank for the model technique.
* This model is intended for developers that want to build IDS system.

### Name who is intended to benefit from this model. 
* Intended for users that use netflow activities to defend against intrusion detection attack.

### Describe the model output.
* This model outputs anomalous score of netflow activities, with large score indicate as suspicious attack.
### List the steps explaining how this model works. (e.g., )  
* Loda detects anomalies in a dataset by computing the likelihood of data points using an ensemble of one-dimensional histograms. These histograms serve as density estimators by approximating the joint probability of the data using sparse random projections<br>

### Name the adversely impacted groups (protected classes) this has been tested to deliver comparable outcomes regardless of:
* Not Applicable

### List the technical limitations of the model.
* This model requires feature engineered netflow activity data in the format of CIC dataset.

### What performance metrics were used to affirm the model's performance?
* AUC & average precision score

### What are the potential known risks to users and stakeholders? 
* Not Applicable

### What training is recommended for developers working with this model?  If none, please state "none."
* none
### Link the relevant end user license agreement 
* [Apache 2.0](https://github.com/nv-morpheus/Morpheus/blob/branch-23.07/LICENSE)

## Model Card ++ Saftey & Security Subcard

### Link the location of the training dataset's repository (if able to share).
* [CICIDS2017]('(https://www.unb.ca/cic/datasets/ids-2017.html))

### Is the model used in an application with physical safety impact?
* No

### Describe physical safety impact (if present).
* Not Applicable

### Was model and dataset assessed for vulnerability for potential form of attack?
* No
### Name applications for the model.
* Typically used to identify abnormality out of Netflow activities
### Name use case restrictions for the model.
* The model is trained in the format of CIC dataset schema, however LODA is generic anomaly detector and can be used for any type of numeric dataset.
### Has this been verified to have met prescribed quality standards?
* No

### Name target quality Key Performance Indicators (KPIs) for which this has been tested.  
* Not Applicable
### Technical robustness and model security validated?
* Not Applicable
### Is the model and dataset compliant with National Classification Management Society (NCMS)?
* Not Applicable

##  Are there explicit model and dataset restrictions?
* No

### Are there access restrictions to systems, model, and data?
* No
### Is there a digital signature?
* No

## Model Card ++ Privacy Subcard


### Generatable or reverse engineerable personally-identifiable information (PII)?

* Neither

### Was consent obtained for any PII used?
* Not Applicable

### Protected classes used to create this model? (The following were used in model the model's training:)

* None of the Above


### How often is dataset reviewed?
* Other: Not Applicable

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
* Yes, for data collected by NVIDIA.  No, for all externally-sourced data.