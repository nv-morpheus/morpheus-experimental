## Industrial Control System (ICS) Cyber Attack Detection

## Use Case
Classify events into various categories based on power system data.

### Version
1.0

### Model Overview
The model is an XGBoost classifier that predicts each event on a power system based on dataset features.

### Model Architecture
XGBoost Classifier

### Requirements
Requirements can be installed with 
```
pip install -r requirements.txt
```
and for `p7zip`
```
apt update
apt install p7zip-full p7zip-rar
```

### Training

#### Training data
In this project, we use the publicly available __[**Industrial Control System (ICS) Cyber Attack Datasets**](Tommy Morris - Industrial Control System (ICS) Cyber Attack Datasets (google.com))__[1] dataset from the Oak Ridge National Laboratories (ORNL) and UAH. We use the 3-class version of the dataset. The dataset labels are Natural Events, No Events and Attack Events. 
Dataset features contain synchrophasor measurements and data logs from Snort, a simulated control panel, and relays. There are 78377 rows in the dataset. In our notebooks and scripts, we download the compressed version from its source and then extract and merge all the rows into a dataframe.

#### Training parameters

Most of the default XGBoost parameters are used in training code. The performance could be improved by finding better hyperparameters. We experimented with a random search but excluded that part from the notebook for brevity.
i.e.
```
params = { 'max_depth': [2,3,6,10,20],
           'learning_rate': [0.05,0.1, 0.15,0.2,0.25,0.3],
           'n_estimators': [500, 750, 1000,1200],
           'colsample_bytree': [0.1,0.3,0.5, 0.7,0.9],
           'min_child_weight': [1, 2, 5,8, 10,12,15,20],
           'gamma': [0.5, 0.75,1, 1.5, 2, 5 , 7,8, 10,12],
           'subsample': [0.05,0.1,0.3,0.6, 0.8, 1.0],
           'colsample_bytree': [0.05,0.1,0.3,0.6, 0.8, 1.0],
         }
scorer={'f1_score' : make_scorer(f1_score, average='weighted')}
grid=RandomizedSearchCV(xgb_clf,params,cv=kfold,random_state=2,scoring=scorer,refit=False,n_iter=40)
```

The hyperparameter set below came up as the best combination; different experiments may give different results.
```
{'subsample': 0.8, 'n_estimators': 1200, 'min_child_weight': 2, 'max_depth': 20, 'learning_rate': 0.15, 'gamma': 0.5, 'colsample_bytree': 0.1}
```

#### Model accuracy

The label distribution in the dataset is not imbalanced, so we do not use the accuracy score. Instead, we use F1 weighted as the metric. The F1 score was over 0.91 on a test set.


#### Training script

To train the model, run the following script:

```
python ot-xgboost-train.py \
    --model ../models/ot-xgboost-20230207.pkl
```
This will download the data (if it is not present) and train a model with a training set, and it will save a model under the `models` directory.

### Inference

Inference script can be run as:
```
python ot-xgboost-inference.py \
    --model ../models/ot-xgboost-20230207.pkl \
    --output ot-validation-output.jsonlines
```
This will download the dataset, the prediction is performed on the test set, and the output is saved into a file.


### Ethical considerations
N/A

### References
1. https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets
2. http://www.ece.uah.edu/~thm0009/icsdatasets/PowerSystem_Dataset_README.pdf
