## Phishing URL Detection via AppShield

## Use Case
Classify URLs as directing to phishing campaigns or not.

## Owner
Haim Elisha

### Version
1.0

### Model Overview
This model is a binary classifier to label phishing URLs and non-phishing URLs obtained from host process data.

### Model Architecture
This model is an LSTM neural network with a fully connected layer to differentiate between legitimate URLs and phishing URLs. Features are derived both from the structure of the URL and the characters in the URL.

### Requirements 
To run this example, additional requirements must be installed into your environment. A supplementary requirements file has been provided in this example directory.

```bash 
pip install -r requirements.txt
```

### Training
#### Training data
Training data consists of 97K URLs labelled as phishing URLs and 100K URLs labelled as legitimate URLs.  

#### Training epochs
150

#### Training batch size
2000

#### GPU model
V100

#### Model accuracy
precision = 0.995 
recall = 0.55

#### Training script

To train the model run the following script under working directory.
```bash
cd ${MORPHEUS_EXPERIMENTAL_ROOT}/phishing-url-detection/training-tuning

# Run training script and save models

python phishurl-appshield-combined-lstm-dnn.py
```

This saves trained model files under `../models` directory. Then the inference script can load the models for future inferences.

### How To Use This Model
Combined with host data from DOCA AppShield, this model can be used to detect phishing URLs. A training notebook is also included so that users can update the model as more labeled data is collected. This model is based just on the URL: processing the structure of the URL and words in the URL. Many malicious URLs seem legitimate and are impossible to detect with our features, thus the recall is limited. We can improve the model by adding WHOIS (https://who.is/) and VirusTotal (https://www.virustotal.com/) infromation about the URL.

### Input
Snapshots of URL plugins collected from DOCA AppShield

### Output
Processes with URLs classified as phishing or non-phishing

#### Out-of-scope use cases
N/A

### Ethical considerations
N/A

### References
https://github.com/Antimalweb/URLNet
