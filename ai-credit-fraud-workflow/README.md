# <p style="text-align: center;">Graph Neural Network Based Model Building<br>for Credit Card Fraud Detection</p>
<p align="center">
  <img src="./img/Splash.jpg" width="50%"/>
</p>

What is presented in this open-source GitHub repository are exemplars of the
three steps within a larger Credit Card Fraud Detection Workflow.  Those steps
being: (a) data prep, (b) model building, and (c) inference. Those three steps
are presented as independent Jupyter Notebooks and within an orchestrated
workflow.

What this example does not show, and which needs to be highlighted, is the
complexity of scaling the problem. The sample datasets are of trivial single-GPU
size. The NVIDIA RAPIDS suite of AI libraries has been proven to scale while still
providing leading performance.

__Note__: The sample datasets must be downloaded manually (see Setup)


Table of Content
* [Background](./docs/background.md)
* [This Workflow](./docs/workflow.md)
* [Datasets and Data Prep](./docs/datasets.md)
* [Setup](./docs/setup.md)

Executing these examples:
1. Setup your environment or container (see [Setup](./docs/setup.md))
1. Download the datasets (see [Datasets](./docs/datasets.md))
1. Start Jupyter
1. Run the [Notebooks](./docs/run_notebooks.md)
  * Determine which dataset you want (Notebook names are related to a dataset)
  * Run the data pre-processing Notebook
  * Run the GNN training Notebook
  * Run the inference Notebook


### Notebooks need to executed in the correct order
The notebooks need to be executed in the correct order. For a particular dataset, the preprocessing notebook must be executed before the training notebook. Once the training notebook produces models, the inference notebook can be executed to run inference on unseen data.


For example, for the TabFormer dataset, the notebooks need to be executed in the following order -

   - preprocess_Tabformer.ipynb
   - train_gnn_based_xgboost.ipynb
   - inference_gnn_based_xgboost_TabFormer.ipynb

To train a standalone XGBoost model, that doesn't utilize node embedding, run the following two notebooks in the following oder -

  - train_xgboost.ipynb
  - inference_xgboost_TabFormer.ipynb

__Note__: Before executing `train_xgboost.ipynb` and `train_gnn_based_xgboost.ipynb` notebooks, make sure that the right dataset is selected in the second code cell of of the notebooks.

```code
    DATASET = TABFORMER
```

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
