# Protective Layers Against Targeted E-Mail (PlateMail)

See overview and example results [here](https://developer.nvidia.com/blog/generative-ai-and-accelerated-computing-for-spear-phishing-detection/).

# Model Overviews

* Model name: 20231024_phishing_full_synthetic.onnx
* Use case: spear-phishing-detection
* Description: This model uses intent inferenced from email content and applicable behavioral sender sketches to predict and score emails as phishing emails.
* Owner: Shawn Davis
* Version: 0.0.1
* Input: Processed and intent labeled email data
* Output: Spear phishing label and score
* Intended users: email end users
* Intende use cases: To detect (spear) phishing emails
* Out-of-scope use cases: Non-English emails
* Metrics: Accuracy=0.88
* Evaluaion Data: 1627 synthetic emails
* Training Data: 6507 synthetic emails
* Ethical Considerations: N/A
---
## Description:
This model uses intent inferenced from email content and applicable behavioral sender sketches to predict and score emails as phishing emails.  <br>

## Model Architecture: 
**Architecture Type:** Not Applicable (N/A)  <br>
**Network Architecture:** None <br>

## Input: (Enter "None" As Needed)
**Input Format:** Processed email with sender data and intents <br>
**Input Parameters:** None <br>
**Other Properties Related to Output:** None <br>

## Output: (Enter "None" As Needed)
**Output Format:** Spear phishing label and score <br>
**Output Parameters:** None <br>
**Other Properties Related to Output:** None <br> 

## Software Integration:
**Runtime(s):** Not Applicable (N/A) <br>

**Supported Hardware Platform(s):** <br>
* [All] <br>

**Supported Operating System(s):** <br>
* [Linux] <br>

## Model Version(s): 
20231024  <br>

# Training & Evaluation: 

## Training Dataset:

**Link:** PIC: Shawn Davis <br>
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** 6507 processed emails from a collection of AI generated emails <br>
**Dataset License:** None <br>

## Evaluation Dataset:
**Link:** PIC: Shawn Davis <br>
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** 1627 processed emails from a collection of AI generated emails <br>
**Dataset License:** None <br>

## Inference:
**Engine:** Other (Not Listed) <br>
**Test Hardware:** <br>
* [Other (Not Listed)]  <br>

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
* This model is intended to be used in the detection of spear phishing emails for email end-users.

### Fill in the blank for the model technique.
* This model is intended for developers that want to build and/or customize spear phishing detection models.

### Name who is intended to benefit from this model. 
* This model is intended for all email users that wish to mitigate potential damages from spear phishing emails.


### Describe the model output. (e.g., This model _____________.)
* This model outputs a spear phishing label and score between 0 and 1 for incoming emails.

### List the steps explaining how this model works. (e.g., )  
* This model uses data processed from an incoming email which includes the text of the email as well as sender metadata and arrival timings. Further, malicious intents are labeled within the email and used as predictors in the final classifciation. This sender behavioral metadata as well as the intents can be returned to the user to illuminate why an email was labeled as spear phishing or not.

### Name the adversely impacted groups (protected classes) this has been tested to deliver comparable outcomes regardless of:
* Not Applicable

### List the technical limitations of the model. (e.g., Technical limitations include: ______________________ (Only If Given))

### What performance metrics were used to affirm the model's performance? (e.g., _______________ (Only the Measure If Given- No Numbers))
* Accuracy and F1

### What are the potential known risks to users and stakeholders? (e.g., Potential risks may include ________________ (fill in the blank)) 
* Potential risks may include unacceptable false positive rates for spear phishing emails.

### What training is recommended for developers working with this model?  If none, please state "none."
* None

### Link the relevant end user license agreement 
*


## Model Card ++ Saftey & Security Subcard

### Link the location of the training dataset's repository (if able to share).
*

### Is the model used in an application with physical safety impact?
* No

### Describe physical safety impact (if present).
* N/A

### Was model and dataset assessed for vulnerability for potential form of attack?
* No

### Name applications for the model.

### Name use case restrictions for the model.
* The model currently only works for English emails.

### Has this been verified to have met prescribed quality standards?
* Yes
* No
* Other:

### Name target quality Key Performance Indicators (KPIs) for which this has been tested.  

### Technical robustness and model security validated?
* Yes
* No
* Other:

* Link the bugs related to technical robustness and model security.


### Is the model and dataset compliant with National Classification Management Society (NCMS)?
* Yes
* No
* In-Progress
* Link NCMS documentation.

### Are there explicit model and dataset restrictions?
* Yes
* No
* In-Progress
* Name explicit model and dataset restrictions.

### Are there access restrictions to systems, model, and data?
* Yes
* No
* In-Progress
* Name the access restrictions.

### Is there a digital signature?
* Yes, this is encrypted.
* No



## Model Card ++ Privacy Subcard


### Generatable or reverse engineerable personally-identifiable information (PII)?
* Neither

### Was consent obtained for any PII used?
* N/A

### Protected classes used to create this model? (The following were used in model the model's training:)
* None of the Above


### How often is dataset reviewed?
* Before Every Release
* Quarterly
* Annually
* Other:

### Is a mechanism in place to honor data subject right of access or deletion of personal data?
* Yes
* No

* For disclosures, name who should be contacted. 

### If PII collected for the development of this AI model, was it minimized to only what was required? 
* N/A

### Is data in dataset traceable?
* Yes
* No

### Scanned for malware?
* Yes
* No
* Link the Legal bugs (or nSpect ID if none).

### Are we able to identify and trace source of dataset?
* Yes

### Does data labeling (annotation, metadata) comply with privacy laws?
* Yes
* No

### Is data compliant with data subject requests for data correction or removal, if such a request was made?
* Yes
* Yes, for data collected by NVIDIA.  No, for all externally-sourced data.
* No

---
# Documentation/ Model Card Info

* Model name: moneyv2_checkpoint-2167
* Use case: spear-phishing-detection
* Description: Detects 'asks for money' in email text.
* Owner: Gorkem Batmaz
* Version: 0.0.1
* Input: Email text
* Output: Intent label and score
* Training epochs: 1
* Training Batch size: 16
* Training GPU: Tesla V100
* Intended Users: N/A; used in pipeline
* Intended Use cases: Label intents for spear phishing classification
* Out-of-scope use cases: Non-English emails
* Metrics: Accuracy=0.99
* Evaluaion Data: 10834 emails
* Training Data: 54167 emails
* Ethical Considerations: N/A

# Model Card ++
# Model Overview

## Description:
This model labels the intent 'asks for money' inferenced from email content.<br>

## Model Architecture: 
**Architecture Type:** transformer  <br>
**Network Architecture:** None <br>

## Input: (Enter "None" As Needed)
**Input Format:** Email text <br>
**Input Parameters:** None <br>
**Other Properties Related to Output:** None <br>

## Output: (Enter "None" As Needed)
**Output Format:** Intent label and score <br>
**Output Parameters:** None <br>
**Other Properties Related to Output:** None <br> 

## Software Integration:
**Runtime(s):** Not Applicable (N/A) <br>

**Supported Hardware Platform(s):** <br>
* [Other (Not Listed)] <br>

**Supported Operating System(s):** <br>
* [Linux] <br>

## Model Version(s): 
moneyv2_checkpoint-2167  <br>

# Training & Evaluation: 

## Training Dataset:

**Link:** PIC: Gorkem Batmaz <br>
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** 54167 emails from a collection of  AI generated emails <br>
**Dataset License:** None <br>

## Evaluation Dataset:
**Link:** PIC: Gorkem Batmaz <br>
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** 10834 emails from a collection of AI generated emails <br>
**Dataset License:** None <br>

## Inference:
**Engine:** Triton <br>
**Test Hardware:** <br>
* [Other (Not Listed)]  <br>

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
* This model is intended to be used in the detection of the presence of specific intents in email text.

### Fill in the blank for the model technique.
* This model is intended for developers that want to build and/or customize spear phishing detection models.

### Name who is intended to benefit from this model. 
* This model is intended for all email users that wish to mitigate potential damages from spear phishing emails.

### Describe the model output. (e.g., This model _____________.)
* This model outputs an intent label and score between 0 and 1 for incoming emails.

### List the steps explaining how this model works. (e.g., )  
* This model uses a distilBERT transformer to detect the presence of a specific intent in an email that is associated with spear phishing attacks.

### Name the adversely impacted groups (protected classes) this has been tested to deliver comparable outcomes regardless of:
* Not Applicable

### List the technical limitations of the model. (e.g., Technical limitations include: ______________________ (Only If Given))

### What performance metrics were used to affirm the model's performance? (e.g., _______________ (Only the Measure If Given- No Numbers))
* Accuracy

### What are the potential known risks to users and stakeholders? (e.g., Potential risks may include ________________ (fill in the blank)) 
* Potential risks may include unacceptable false positive and false negative rates for detecting intent.

### What training is recommended for developers working with this model?  If none, please state "none."
* None

### Link the relevant end user license agreement 
*


## Model Card ++ Saftey & Security Subcard

### Link the location of the training dataset's repository (if able to share).
*

### Is the model used in an application with physical safety impact?
* No

### Describe physical safety impact (if present).
* N/A

### Was model and dataset assessed for vulnerability for potential form of attack?
* No

### Name applications for the model.

### Name use case restrictions for the model.
* The model currently only works for English emails.

### Has this been verified to have met prescribed quality standards?
* Yes
* No
* Other:

### Name target quality Key Performance Indicators (KPIs) for which this has been tested.  

### Technical robustness and model security validated?
* Yes
* No
* Other:

* Link the bugs related to technical robustness and model security.


### Is the model and dataset compliant with National Classification Management Society (NCMS)?
* Yes
* No
* In-Progress
* Link NCMS documentation.

### Are there explicit model and dataset restrictions?
* Yes
* No
* In-Progress
* Name explicit model and dataset restrictions.

### Are there access restrictions to systems, model, and data?
* Yes
* No
* In-Progress
* Name the access restrictions.

### Is there a digital signature?
* Yes, this is encrypted.
* No



## Model Card ++ Privacy Subcard


### Generatable or reverse engineerable personally-identifiable information (PII)?
* Neither

### Was consent obtained for any PII used?
* N/A

### Protected classes used to create this model? (The following were used in model the model's training:)
* None of the Above


### How often is dataset reviewed?
* Before Every Release
* Quarterly
* Annually
* Other:

### Is a mechanism in place to honor data subject right of access or deletion of personal data?
* Yes
* No

* For disclosures, name who should be contacted. 

### If PII collected for the development of this AI model, was it minimized to only what was required? 
* N/A

### Is data in dataset traceable?
* Yes
* No

### Scanned for malware?
* Yes
* No
* Link the Legal bugs (or nSpect ID if none).

### Are we able to identify and trace source of dataset?
* Yes

### Does data labeling (annotation, metadata) comply with privacy laws?
* Yes
* No

### Is data compliant with data subject requests for data correction or removal, if such a request was made?
* Yes
* Yes, for data collected by NVIDIA.  No, for all externally-sourced data.
* No
---
* Model name: personalv2_checkpoint-2167
* Use case: spear-phishing-detection
* Description: Detects 'asks for personal information' in email text, e.g. bank account information, Social Security Number.
* Owner: Gorkem Batmaz
* Version: 0.0.1
* Input: Email text
* Output: Intent label and score
* Training epochs: 1
* Training Batch size: 16
* Training GPU: Tesla V100
* Intended Users: N/A; used in pipeline
* Intended Use cases: Label intents for spear phishing classification
* Out-of-scope use cases: Non-English emails
* Metrics: Accuracy=0.99
* Evaluaion Data: 10834 emails
* Training Data: 54167 emails
* Ethical Considerations: N/A

# Model Card ++
# Model Overview

## Description:
This model labels the intent 'asks for personal information' inferenced from email content.<br> 

## Model Architecture: 
**Architecture Type:** transformer  <br>
**Network Architecture:** None <br>

## Input: (Enter "None" As Needed)
**Input Format:** Email text <br>
**Input Parameters:** None <br>
**Other Properties Related to Output:** None <br>

## Output: (Enter "None" As Needed)
**Output Format:** Intent label and score <br>
**Output Parameters:** None <br>
**Other Properties Related to Output:** None <br> 

## Software Integration:
**Runtime(s):** Not Applicable (N/A) <br>

**Supported Hardware Platform(s):** <br>
* [Other (Not Listed)] <br>

**Supported Operating System(s):** <br>
* [Linux] <br>

## Model Version(s): 
personalv2_checkpoint-2167  <br>

# Training & Evaluation: 

## Training Dataset:

**Link:** PIC: Gorkem Batmaz <br>
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** 54167 emails from a collection of AI generated emails <br>
**Dataset License:** None <br>

## Evaluation Dataset:
**Link:** PIC: Gorkem Batmaz <br>
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** 10834 emails from a collection of AI generated emails <br>
**Dataset License:** None <br>

## Inference:
**Engine:** Triton <br>
**Test Hardware:** <br>
* [Other (Not Listed)]  <br>

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
* This model is intended to be used in the detection of the presence of specific intents in email text.

### Fill in the blank for the model technique.
* This model is intended for developers that want to build and/or customize spear phishing detection models.

### Name who is intended to benefit from this model. 
* This model is intended for all email users that wish to mitigate potential damages from spear phishing emails.

### Describe the model output. (e.g., This model _____________.)
* This model outputs an intent label and score between 0 and 1 for incoming emails.

### List the steps explaining how this model works. (e.g., )  
* This model uses a distilBERT transformer to detect the presence of a specific intent in an email that is associated with spear phishing attacks.

### Name the adversely impacted groups (protected classes) this has been tested to deliver comparable outcomes regardless of:
* Not Applicable

### List the technical limitations of the model. (e.g., Technical limitations include: ______________________ (Only If Given))

### What performance metrics were used to affirm the model's performance? (e.g., _______________ (Only the Measure If Given- No Numbers))
* Accuracy

### What are the potential known risks to users and stakeholders? (e.g., Potential risks may include ________________ (fill in the blank)) 
* Potential risks may include unacceptable false positive and false negative rates for detecting intent.

### What training is recommended for developers working with this model?  If none, please state "none."
* None

### Link the relevant end user license agreement 
*


## Model Card ++ Saftey & Security Subcard

### Link the location of the training dataset's repository (if able to share).
*

### Is the model used in an application with physical safety impact?
* No

### Describe physical safety impact (if present).
* N/A

### Was model and dataset assessed for vulnerability for potential form of attack?
* No

### Name applications for the model.

### Name use case restrictions for the model.
* The model currently only works for English emails.

### Has this been verified to have met prescribed quality standards?
* Yes
* No
* Other:

### Name target quality Key Performance Indicators (KPIs) for which this has been tested.  

### Technical robustness and model security validated?
* Yes
* No
* Other:

* Link the bugs related to technical robustness and model security.


### Is the model and dataset compliant with National Classification Management Society (NCMS)?
* Yes
* No
* In-Progress
* Link NCMS documentation.

### Are there explicit model and dataset restrictions?
* Yes
* No
* In-Progress
* Name explicit model and dataset restrictions.

### Are there access restrictions to systems, model, and data?
* Yes
* No
* In-Progress
* Name the access restrictions.

### Is there a digital signature?
* Yes, this is encrypted.
* No



## Model Card ++ Privacy Subcard


### Generatable or reverse engineerable personally-identifiable information (PII)?
* Neither

### Was consent obtained for any PII used?
* N/A

### Protected classes used to create this model? (The following were used in model the model's training:)
* None of the Above


### How often is dataset reviewed?
* Before Every Release
* Quarterly
* Annually
* Other:

### Is a mechanism in place to honor data subject right of access or deletion of personal data?
* Yes
* No

* For disclosures, name who should be contacted. 

### If PII collected for the development of this AI model, was it minimized to only what was required? 
* N/A

### Is data in dataset traceable?
* Yes
* No

### Scanned for malware?
* Yes
* No
* Link the Legal bugs (or nSpect ID if none).

### Are we able to identify and trace source of dataset?
* Yes

### Does data labeling (annotation, metadata) comply with privacy laws?
* Yes
* No

### Is data compliant with data subject requests for data correction or removal, if such a request was made?
* Yes
* Yes, for data collected by NVIDIA.  No, for all externally-sourced data.
* No
---
# Documentation/ Model Card Info

* Model name: crypto_checkpoint-2362
* Use case: spear-phishing-detection
* Description: Detects 'crypto based scam' in email text.
* Owner: Gorkem Batmaz
* Version: 0.0.1
* Input: Email text
* Output: Intent label and score
* Training epochs: 1
* Training Batch size: 16
* Training GPU: Tesla V100
* Intended Users: N/A; used in pipeline
* Intended Use cases: Label intents for spear phishing classification
* Out-of-scope use cases: Non-English emails
* Metrics: Accuracy=0.99
* Evaluaion Data: 1773 emails
* Training Data: 7905 emails
* Ethical Considerations: N/A

# Model Card ++
# Model Overview

## Description:
This model labels the intent 'crypto based scam' inferenced from email content.<br>


## Model Architecture: 
**Architecture Type:** transformer  <br>
**Network Architecture:** None <br>

## Input: (Enter "None" As Needed)
**Input Format:** Email text <br>
**Input Parameters:** None <br>
**Other Properties Related to Output:** None <br>

## Output: (Enter "None" As Needed)
**Output Format:** Intent label and score <br>
**Output Parameters:** None <br>
**Other Properties Related to Output:** None <br> 

## Software Integration:
**Runtime(s):** Not Applicable (N/A) <br>

**Supported Hardware Platform(s):** <br>
* [Other (Not Listed)] <br>

**Supported Operating System(s):** <br>
* [Linux] <br>

## Model Version(s): 
moneyv2_checkpoint-2167  <br>

# Training & Evaluation: 

## Training Dataset:

**Link:** PIC: Gorkem Batmaz <br>
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** 7905 emails from a collection of AI generated emails <br>
**Dataset License:** None <br>

## Evaluation Dataset:
**Link:** PIC: Gorkem Batmaz <br>
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** 1773 emails from a collection of AI generated emails <br>
**Dataset License:** None <br>

## Inference:
**Engine:** Triton <br>
**Test Hardware:** <br>
* [Other (Not Listed)]  <br>

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
* This model is intended to be used in the detection of the presence of specific intents in email text.

### Fill in the blank for the model technique.
* This model is intended for developers that want to build and/or customize spear phishing detection models.

### Name who is intended to benefit from this model. 
* This model is intended for all email users that wish to mitigate potential damages from spear phishing emails.

### Describe the model output. (e.g., This model _____________.)
* This model outputs an intent label and score between 0 and 1 for incoming emails.

### List the steps explaining how this model works. (e.g., )  
* This model uses a distilBERT transformer to detect the presence of a specific intent in an email that is associated with spear phishing attacks.

### Name the adversely impacted groups (protected classes) this has been tested to deliver comparable outcomes regardless of:
* Not Applicable

### List the technical limitations of the model. (e.g., Technical limitations include: ______________________ (Only If Given))

### What performance metrics were used to affirm the model's performance? (e.g., _______________ (Only the Measure If Given- No Numbers))
* Accuracy

### What are the potential known risks to users and stakeholders? (e.g., Potential risks may include ________________ (fill in the blank)) 
* Potential risks may include unacceptable false positive and false negative rates for detecting intent.

### What training is recommended for developers working with this model?  If none, please state "none."
* None

### Link the relevant end user license agreement 
*


## Model Card ++ Saftey & Security Subcard

### Link the location of the training dataset's repository (if able to share).
*

### Is the model used in an application with physical safety impact?
* No

### Describe physical safety impact (if present).
* N/A

### Was model and dataset assessed for vulnerability for potential form of attack?
* No

### Name applications for the model.

### Name use case restrictions for the model.
* The model currently only works for English emails.

### Has this been verified to have met prescribed quality standards?
* Yes
* No
* Other:

### Name target quality Key Performance Indicators (KPIs) for which this has been tested.  

### Technical robustness and model security validated?
* Yes
* No
* Other:

* Link the bugs related to technical robustness and model security.


### Is the model and dataset compliant with National Classification Management Society (NCMS)?
* Yes
* No
* In-Progress
* Link NCMS documentation.

### Are there explicit model and dataset restrictions?
* Yes
* No
* In-Progress
* Name explicit model and dataset restrictions.

### Are there access restrictions to systems, model, and data?
* Yes
* No
* In-Progress
* Name the access restrictions.

### Is there a digital signature?
* Yes, this is encrypted.
* No



## Model Card ++ Privacy Subcard


### Generatable or reverse engineerable personally-identifiable information (PII)?
* Neither

### Was consent obtained for any PII used?
* N/A

### Protected classes used to create this model? (The following were used in model the model's training:)
* None of the Above


### How often is dataset reviewed?
* Before Every Release
* Quarterly
* Annually
* Other:

### Is a mechanism in place to honor data subject right of access or deletion of personal data?
* Yes
* No

* For disclosures, name who should be contacted. 

### If PII collected for the development of this AI model, was it minimized to only what was required? 
* N/A

### Is data in dataset traceable?
* Yes
* No

### Scanned for malware?
* Yes
* No
* Link the Legal bugs (or nSpect ID if none).

### Are we able to identify and trace source of dataset?
* Yes

### Does data labeling (annotation, metadata) comply with privacy laws?
* Yes
* No

### Is data compliant with data subject requests for data correction or removal, if such a request was made?
* Yes
* Yes, for data collected by NVIDIA.  No, for all externally-sourced data.
* No