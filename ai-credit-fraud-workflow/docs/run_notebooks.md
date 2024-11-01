# Running the Notebooks
This page will go over the sequence to run the various notebooks.  
Please note that once the data is prepared, both datasets leverage the same notebooks for training.

__Note:__ It is assumed that the data has been downloaded and placed in the raw folder for each respective dataset.
if not, please see: [setup](./setup.md)

__Note__:It is also assumed that Jupyter has been started and the conda environment has been added. See [setup](./setup.md)

__Note__: Before executing `train_xgboost.ipynb` and `train_gnn_based_xgboost.ipynb` notebooks, make sure that the right dataset is selected in the second code cell of of the notebooks.

For TabFormer dataset, set
```code
    DATASET = TABFORMER
```
and for the Sparkov dataset, set
```code
    DATASET = SPARKOV
```

## TabFormer

### Step 1: Prepare the data
run `notebooks/preprocess_Tabformer.ipynb`

This will produce a number of files under `./data/TabFormer/gnn` and `./data/TabFormer/xgb`. It will also save data preprocessor pipeline `preprocessor.pkl` and a few variables in a json file `variables.json` under `./data/TabFormer` directory.

### Step 2: Build the model
run `notebooks/train_gnn_based_xgboost.ipynb`

This will produce two files for the GNN-based XGBoost model under `./data/TabFormer/models` directory.

### Step 3: Run Inference
run `notebooks/inference_gnn_based_xgboost_TabFormer.ipynb`

### Optional: Pure XGBoost
Two additional notebooks are provided to build a pure XGBoost model (without GNN) and perform inference using that model.

__Train__
run `notebooks/train_xgboost.ipynb`

This will produce a XGBoost model under `./data/TabFormer/models` directory.

__Inference__
run `notebooks/inference_xgboost_TabFormer.ipynb`



## Sparkov

__Note__ Make sure to restart jupyter kernel before running `train_gnn_based_xgboost.ipynb` for the second dataset.

### Step 1: Prepare the data
run `notebooks/preprocess_Sparkov.ipynb`

This will produce a number of files under `./data/Sparkov/gnn` and `./data/Sparkov/xgb`. It will also save data preprocessor pipeline `preprocessor.pkl` and a few variables in a json file `variables.json` under `./data/Sparkov` directory.

### Step 2: Build the model
run `notebooks/train_gnn_based_xgboost.ipynb`

This will produce two files for the GNN-based XGBoost model under `./data/Sparkov/models` directory.


### Optional: Pure XGBoost
Two additional notebooks are provided to build a pure XGBoost model (without GNN) and perform inference using that model.

__Train__
run `notebooks/train_xgboost.ipynb`

This will produce a XGBoost model under `./data/Sparkov/models` directory.

__Inference__
run `notebooks/inference_xgboost_Sparkov.ipynb`


<br/>
<hr/>

[<-- Top](../README.md) </br>

<br/><br/>

## Copyright and License
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

<br/>

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
 http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
