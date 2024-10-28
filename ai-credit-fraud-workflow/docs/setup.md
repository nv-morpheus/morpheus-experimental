# Setup
There are a number of ways that the notebooks can be executed.  



## Step 1: Clone the repo

cd into the base directory where you pkan to house the code.

```bash
git clone https://github.com/nv-morpheus/morpheus-experimental
cd ./morpheus-experimental/ai-credit-fraud-workflow
```

## Step 2: Download the datasets

__TabFormer__</br>
1. Download the dataset: https://ibm.ent.box.com/v/tabformer-data/folder/130747715605
2. untar and uncompreess the file:  `tar -xvzf ./transactions.tgz`
3. Place the file in the ___"./data/TabFormer/raw"___ folder 


__Sparkov__</br>
1. Download the dataset from: https://www.kaggle.com/datasets/kartik2112/fraud-detection
2. Unzip the "archive.zip" file
    * that will produce a folder with two files
3. place the two files under the __"./data/'Sparkov/raw"__ folder

## Step 3: Create a new conda environment

You can get a minimum installation of Conda and Mamba using [Miniforge](https://github.com/conda-forge/miniforge).

And then create an environment using the following command.

Make sure that your shell or command prompt is pointint to `morpheus-experimental/ai-credit-fraud-workflow` before running `mamba env create`.

```bash
~/morpheus-experimental/ai-credit-fraud-workflow$ mamba env create -f conda/fraud_conda_env.yaml
```


Alternatively, you can install [MiniConda](https://docs.anaconda.com/miniconda/miniconda-install) and run the following commands to create an environment to run the notebooks.

 Install `mamba` first with

```bash
conda install conda-forge::mamba
```
And, then run `mamba env create` from the right directory as shown below.

```bash
~/morpheus-experimental/ai-credit-fraud-workflow$ mamba env create -f conda/fraud_conda_env.yaml
```

Finally, activate the environment.

```bash
conda activate fraud_conda_env
```

All the notebooks are located under `morpheus-experimental/ai-credit-fraud-workflow/notebooks`.

```bash
~/morpheus-experimental/ai-credit-fraud-workflow$ cd notebooks
~/morpheus-experimental/ai-credit-fraud-workflow/notebooks$ ls -1
inference_gnn_based_xgboost_TabFormer.ipynb
inference_xgboost_Sparkov.ipynb
inference_xgboost_TabFormer.ipynb
preprocess_Sparkov.ipynb
preprocess_Tabformer.ipynb
train_gnn_based_xgboost.ipynb
train_xgboost.ipynb
```

Now you can run the notebooks from VS Code. Note that you need to select `fraud_conda_env` as the kernel in VS Code to run the notebooks. Alternatively, you can run the notebooks using Jupyter or Jupyter labs.  You will need to add the conda environment:  `ipython kernel install --user --name= fraud_conda_env`


#### NOTE: Notebooks need to be executed in the correct order
The notebooks need to be executed in the correct order. For a particular dataset, the preprocessing notebook must be executed before the training notebook. Once the training notebook produces models, the inference notebook can be executed to run inference on unseen data.

For example, for the TabFormer dataset, the notebooks need to be executed in the following order -

   - preprocess_Tabformer.ipynb
   - train_gnn_based_xgboost.ipynb
   - inference_gnn_based_xgboost_TabFormer.ipynb

The train a standalone XGBoost model, that doesn't utilize node embedding, run the following two notebooks in the following oder -

  - train_xgboost.ipynb
  - inference_xgboost_TabFormer.ipynb

## Docker container (alternative,to creating a conda environment)

If you don't want to create a conda environment locally, you can spin up a Docker container either on your local machine or a remote one and execute the notebooks from a browser or the terminal.

### Running locally

Clone the [repo](https://github.com/nv-morpheus/morpheus-experimental) and `cd` into the project folder
```bash
git clone https://github.com/nv-morpheus/morpheus-experimental
cd morpheus-experimental/ai-credit-fraud-workflow
```


### Build docker image and run a container with port forwarding

Build the docker image from `morpheus-experimental/ai-credit-fraud-workflow`
```bash
~/morpheus-experimental/ai-credit-fraud-workflow$ docker build --no-cache -t fraud-detection-app .
```

And, run a container from `morpheus-experimental/ai-credit-fraud-workflow`

```bash
~/morpheus-experimental/ai-credit-fraud-workflow$ docker run --gpus all -it --rm -v $(pwd):/ai-credit-fraud-workflow -p 8888:8888 fraud-detection-app
```

This will give you an interactive shell into the docker container. All the notebooks should be accessible under `/ai-credit-fraud-workflow/notebooks` inside the container.

__Note__: `-v $(pwd):/ai-credit-fraud-workflow` in the `docker run` command will mount `morpheus-experimental/ai-credit-fraud-workflow` directory from the host machine into the Docker container as `/ai-credit-fraud-workflow`.

You can list the notebooks from the interactive shell of the docker container. Note that you will have a different container id than shown (7c593d76f681) in the example output below.

```bash
root@7c593d76f681:/ai-credit-fraud-workflow# ls
Dockerfile  LICENSE  README.md  conda  data  docs  img  notebooks  python  requirements.txt

root@7c593d76f681:/ai-credit-fraud-workflow# cd notebooks/

root@7c593d76f681:/ai-credit-fraud-workflow/notebooks# ls -1
inference_gnn_based_xgboost_TabFormer.ipynb
inference_xgboost_Sparkov.ipynb
inference_xgboost_TabFormer.ipynb
preprocess_Sparkov.ipynb
preprocess_Tabformer.ipynb
train_gnn_based_xgboost.ipynb
train_xgboost.ipynb
```

### Launch Jupyter Notebook inside the container

Run the following command from interactive shell inside the docker container.
```bash
root@7c593d76f681:/ai-credit-fraud-workflow# jupyter notebook .
```
It will display an url with token
http://127.0.0.1:8888/tree?token=<token_value>

Now you can browse to the `notebooks` folder, and run or edit the notebooks from a browser at the url.


If you are not interested in running/editing the notebooks from a browser, you can omit the port forwarding option.

```bash
~/morpheus-experimental/ai-credit-fraud-workflow$ docker build --no-cache -t fraud-detection-app .
```

```bash
~/morpheus-experimental/ai-credit-fraud-workflow$ docker run --gpus all -it --rm -v $(pwd):/ai-credit-fraud-workflow fraud-detection-app
```

This will give you an interactive shell inside the docker container.

And then, you can run any notebook using the following command inside the container.

```bash
root@7c593d76f681:/ai-credit-fraud-workflow# cd notebooks
root@7c593d76f681:/ai-credit-fraud-workflow# jupyter nbconvert --to notebook --execute [NAME_OF_THE_NOTBOOK].ipynb --output [NAME_OF_THE_OUTPUT_NOTEBOOK].ipynb
```

### Running on a remote machine

### Copy the dataset to the right folder

```bash
scp path/to/downloaded-file-in your-local-machine  user@remote_host_name:path/to/ai-credit-fraud-workflow/data/[DATASET_NAME]/raw
```

Make sure to place the unzipped csv file in  `ai-credit-fraud-workflow/data/[DATASET_NAME]/raw` folder.


Login to your remote machine from your host machine, with ssh tunneling/port forwarding

```bash
ssh -L 127.0.0.1:8888:127.0.0.1:8888 USER@REMOTE_HOST_NAME_OR_IP
```

Then follow the steps described under section `Launch Jupyter Notebook inside the container` . Finally, go to the url from a browser in your host machine to run/edit the notebooks.

[<-- Top](../README.md) </br>
[<-- Back: Datasets](./datasets.md) </br>
