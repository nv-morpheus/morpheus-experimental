# Datasets
The exemplars here are based on two different datasets with a different set of notebooks for each dataset.  

__Both datasets need to be download manually.__

## Dataset 1: IBM TabFormer
* https://github.com/IBM/TabFormer
  * just the Credit Card Transaction Dataset and not the others
* License:  Apache License Version 2.0, January 2004
* 24 million transaction records


## Dataset 2: Sparkov
The data generator:
  * https://github.com/namebrandon/Sparkov_Data_Generation


The generator was used to produce a dataset for Kaggle:
  * https://www.kaggle.com/datasets/kartik2112/fraud-detection 
  * Released under CC0: Public Domain
  * Contains 1,296,675 records with 23 fields
    * one field being the "is_fraud" label which we use for training.


<br/><br/>


# Data Prep

Preprocessing, along with feature engineering are very important steps in machine learning that significantly impact model performance. Here is summary of preprocessing we performed for the two datasets 

## TabFormer

### Data fields
* Ordinal categorical fields - 'Year', 'Month', 'Day'
* Nominal categorical fields - 'User', 'Card', 'Merchant Name', 'Merchant City', 'Merchant State', 'Zip', 'MCC', 'Errors?'
* Target label - 'Is Fraud?'

### Preprocessing
* Missing values for 'Merchant State', 'Zip' and 'Errors?' fields are replaced with markers as these columns have nominal categorical values.
* Dollar symbol ($) in 'Amount'  and extra character (,) in 'Errors?' field are removed.
* 'Time' in converted to number of minutes over the span of a day.
* 'Card' is converted to 'User' * MAX_NUMBER_OF_CARD_PER_USERS + 'Card' and finally treated as nominal categorical values to make sure that Card 0 from User 1 is different from Card 0 of User 2 
* Filtered out categorical and numerical columns that don't have significant correlation with target column
* Hot-encoded nominal categorical columns with less than nine categories and binary encoded nominal categorical columns with nine or more categories
* Scaled numerical column. As the 'Amount' field has a few extreme values, we scaled the field with a Robust Scaler.
* We save the fitted transformer, transformed train and test data in CSV files.
  
NOTE: Binary encoding and scaling performed using a column transformer, which is composed of encoders and a scaler.

### To create Graph from GNN
* Assigned unique and consecutive ids for the transactions, which become node ids of the transactions in the Graph.
* Card (or user) ids are used to create consecutive ids for user nodes
* Merchant strings are converted mapped to consecutive ids for merchant nodes.
* If an user U makes a transaction T to a merchant M, user node U will have an edge (directional or bidirectional depending on flag) to transaction node T, and the transaction node T will be connected with an edge (directional or bidirectional depending on flag) to the merchant node M.
* Transformed transaction node features are saved in a csv file using node id as index.
* Merchant and User nodes are initialized with zero vectors of same length of a transaction node features.
* Target values of all the nodes are saved in a separate CSV file which are loaded during GNN training.


## Sparkov

### Data fields
* Nominal categorical fields - 'cc_num', 'merchant', 'category', 'first', 'last', 'street', 'city', 'state', 'zip', 'job', 'trans_num'
* Numerical fields - 'amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long'
* Timestamp fields - 'dob', 'trans_date_trans_time', 'unix_time'
* Target label - 'is_fraud'

### Preprocessing
* From 'unix_time' and ('lat', 'long') and ('merchant_lat', 'merchant_long') we calculated the transaction 'speed'.
* Converted 'dob' to age.
* Converted 'trans_date_trans_time' in  to number of minutes over the span of a day.
   
* Filter out categorical and numerical columns that don't have significant correlation with target column.
* Binary encoded nominal categorical columns.
* Scaled numerical columns. As the 'amt' field has a few extreme values, we scaled the field with a Robust Scaler. The 'speed' and 'age' are scaled with standard scaler.
* We save the fitted transformer, transformed train and test data in CSV files.
  
NOTE: Binary encoding and scaling performed using a column transformer, which is composed of encoders and scalers.

### To create Graph from GNN
* Assigned unique and consecutive ids for the transactions, which become node ids of the transactions in the Graph.
* 'cc_num' are used to create consecutive ids for user nodes
* Merchant strings are converted mapped to consecutive ids for merchant nodes.
* If an user U makes a transaction T to a merchant M, user node U will have an edge (directional or bidirectional depending on flag) to transaction node T, and the transaction node T will be connected with an edge (directional or bidirectional depending on flag) to the merchant node M.
* Transformed transaction node features are saved in a csv file using node id as index.
* Merchant and User nodes are initialized with zero vectors of same length of a transaction node features.
* Target values of all the nodes are saved in a separate CSV file which are loaded during GNN training.


<br/>
<hr/>

[<-- Top](../README.md) </br>
[<-- Back: Workflow](./workflow.md) </br>
[--> Next: Setup](./setup.md)

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
