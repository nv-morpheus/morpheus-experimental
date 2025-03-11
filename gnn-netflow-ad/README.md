<!--
  SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# GNN-based Autoencoder for Netflow Anomaly Detection Using NVIDIA Morpheus

This example demonstrates how to integrate a Graph Neural Network (GNN) autoencoder architecture into the NVIDIA Morpheus pipeline framework for anomaly detection tasks on Netflow data. By modeling network flows as graphs, this approach can capture complex relationships between hosts, ports, and protocols. The example uses a Graph Autoencoder (GAE) to learn latent representations of benign traffic, enabling the detection of anomalous (malicious) traffic by measuring deviations in reconstruction probabilities.

## Use Case

Modern network environments generate large volumes of Netflow records that describe traffic between sources and destinations. Identifying anomalies in such data is critical for cybersecurity. Traditional methods struggle with the complexity and evolving nature of network traffic. This example addresses this challenge by:

- Representing hosts or devices as nodes and their connections as edges.
- Encoding node and edge attributes to learn a latent space of normal behavior.
- Using the GAE model’s reconstruction of the graph structure as an indicator of anomalous edges.

## Repository Structure

The repository is organized as follows:

```
gnn-netflow-ad/
   ├── morpheus_inference_pipeline.py # Example Morpheus inference pipeline script
   ├── training-tuning.ipynb          # Jupyter notebook for training and tuning the model
   └── src/
       ├── preprocess.py              # Converts raw Netflow into graph data
       ├── guided_gae_model.py        # GAE model architecture
       └── stages/
           ├── connection_source_stage.py
           ├── graph_construction_stage.py
           ├── graph_inference_stage.py
           └── combine_predictions_stage.py
```

### Key Components

- **training-tuning.ipynb:**  
  Walks through data loading, preprocessing, graph construction, GAE training, and evaluation. It uses only benign data for training and evaluates performance on mixed benign and malicious data.

- **inference-pipeline.py:**  
  Demonstrates a Morpheus pipeline that:
  1. Streams Netflow data through a series of stages.
  2. Constructs graphs from incoming edges.
  3. Applies a trained GAE model to determine anomaly scores.
  4. Combines and outputs results.

- **src/preprocess.py:**  
  Preprocessing logic to:
  - Convert Netflow records into PyTorch Geometric `Data` objects.
  - Scale and normalize edge attributes based on benign training data.

- **src/guided_gae_model.py:**  
  Defines the GAE model with global edge embeddings, employing:
  - Graph Attention Layers
  - Graph U-Net for multi-scale feature aggregation
  - Edge attribute embeddings and a global edge context to guide reconstruction

- **Custom Morpheus Stages (src/stages):**  
  Designed to integrate the GNN-based anomaly detection into the NVIDIA Morpheus ecosystem, handling tasks like:
  - Streaming Netflow events.
  - Constructing graphs on the fly.
  - Running inference with the GAE model.
  - Combining predictions into final outputs.

## Dataset

All data used in this example is sourced from the NIDS datasets at the University of Queensland:  
[https://staff.itee.uq.edu.au/marius/NIDS_datasets/](https://staff.itee.uq.edu.au/marius/NIDS_datasets/)

All metrics reported in this documents are based on datasets sourced from above. 

**Sarhan, M., Layeghy, S., Moustafa, N., & Portmann, M.** (2021). *NetFlow Datasets for Machine Learning-Based Network Intrusion Detection Systems*. In **Big Data Technologies and Applications. BDTA 2020, WiCON 2020**. Springer, Cham. [https://doi.org/10.1007/978-3-030-72802-1_9](https://doi.org/10.1007/978-3-030-72802-1_9)


## Training Procedure

1. **Preprocessing:**  
   Load and scale Netflow data. Split it into windows, each represented as a graph.

2. **Graph Construction:**  
   Convert windows of Netflow data into graph structures where nodes represent hosts and edges represent connections.

3. **Model Training:**  
   Train the GAE on benign-only data. The GAE tries to reconstruct the graph’s adjacency structure, including node and edge attributes, from latent embeddings.

4. **Evaluation:**  
   Evaluate the model’s ability to identify malicious edges by checking how well it reconstructs the input. Poor reconstruction indicates anomalies.

## Inference Pipeline

The `inference-pipeline.py` file demonstrates how to integrate the trained model into a Morpheus pipeline for real-time inference:

- **ConnectionSourceStage:** Streams incoming Netflow-like data.
- **GraphConstructionStage:** Builds graphs incrementally as data arrives.
- **GraphInferenceStage:** Applies the trained GAE to infer anomaly scores for each edge.
- **CombinePredictionsStage:** Aggregates predictions to form a final anomaly score or label.

Users can adapt these stages to integrate into larger security analytics pipelines.

**NOTE: Before running the inference pipeline, please save sample data, an edge scaler, and model weights into the `artifacts` directory. You can find instructions on saving these in the training-testing notebook.**

## Deployment

1. **Build the docker container:**
   ```bash
    docker build -f docker/Dockerfile . -t morpheus:gnn-anomaly-detection
   ```
2. **Run the container and start Jupyterlab:**
   ```bash
    docker run --rm -ti --gpus='"device=0"'  --net=host --cap-add=sys_nice \
   -v ${PWD}:/workspace/examples   -v /var/run/docker.sock:/var/run/docker.sock  \
    -w /workspace/examples   --name gnn-anomaly-detection \
    morpheus:gnn-anomaly-detection jupyter-lab --no-browser --allow-root --ip='*'
   ```
   
3. **Running the Inference Pipeline:**
   ```bash
   python3 inference-pipeline.py
   ```
   Adjust pipeline parameters as needed. 

   As benchmarked on a Single `NVIDIA A100-80GBSXM` GPU, and `Morpheus 24.06`, the above pipeline is capable of handling a peak inference throughput of 2.5 million flows/sec.
   

4. **Integration:**

   Modify the input stages or thresholding logic to integrate into a production environment. Adapt data sources and sink stages as required.

If you're using a custom trained model, please modify the `artifacts` configurations in the pipeline to point to your new model weights, scalers, etc.

## Performance Metrics

We track the following metrics to gauge the model’s performance:

- **ROC-AUC:** Measures the model’s discriminative power between benign and malicious edges.
- **Average Precision:** Evaluates how well the model handles class imbalance and captures anomalous edges.
- **Confusion Matrix & Thresholding:** Helps identify trade-offs between false positives and false negatives.
- **Visualizations (ROC Curve, PR Curve):** Provides insight into model performance and aids in fine-tuning detection thresholds.

Below are our compute performance metrics for the on subsets of the NF Intrusion detection datasets. 

| Dataset          | TPR  | FPR  | Num Anomalous Flows | Num Classes | Anomal-E TPR/FPR |
|------------------|------|------|---------------------|------------|-------------------|
| NF-CICIDS-2018   | 87%  | 15%  | 1.1M                | 6          | 88.5%/28.9%       |
| NF-UNSW-NB15     | 98%  | 2%   | 72k                 | 9          | 78.6%/0.18%       |
| NF-ToN-IOT       | 78%  | 4%   | 21k                 | 9          | 73.5%/56.8%       |
| NF-BoT-IOT       | 40%  | 2%   | 586k                | 4          | 45.5%/59.4%       |

**Sarhan, M., Layeghy, S., Moustafa, N., & Portmann, M.** (2021). *NetFlow Datasets for Machine Learning-Based Network Intrusion Detection Systems*. In **Big Data Technologies and Applications. BDTA 2020, WiCON 2020**. Springer, Cham. [https://doi.org/10.1007/978-3-030-72802-1_9](https://doi.org/10.1007/978-3-030-72802-1_9)

**Caville, E., Lo, W. W., Layeghy, S., & Portmann, M.** (2022). *Anomal-E: A self-supervised network intrusion detection system based on graph neural networks.* Knowledge-Based Systems, 258, 110030. [https://doi.org/10.1016/j.knosys.2022.110030](https://doi.org/10.1016/j.knosys.2022.110030)

**Anomal-E benchmark**: [https://gitlab-master.nvidia.com/ads/netflow-anomaly-detection](https://gitlab-master.nvidia.com/ads/netflow-anomaly-detection)
