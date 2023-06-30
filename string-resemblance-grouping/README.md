# String Resemblance Grouping (SRG)
String Resemblance Grouping (SRG) is designed to find a subset of representative strings within a a large collection of messages. These representative strings create groupings with which to categorize the messages for further exploration or triage.

## Version
1.0

## Requirements
SRG requires an environment set-up to use [Rapids](https://rapids.ai/start.html).

## Contents
- [Problem Background](#problem-background)
- [Use Case](#use-case)
- [Technique Overview](#technique-overview)
- [Model Overview](#model-overview)
- [Training Input](#training-input)
- [Inference Input](#inference-input)
- [Inference Output](#inference-output)
- [Future Work](#future-work)
- [References](#references)

### Problem Background

When approaching the problem of categorizing computer logs into groups with an assigned representative, there are two major considerations: the run time of the algorithm and hyperparameter selection. When confronted with millions of log entries with such a large number being unique, the primary approach for many of these data sets is reactive analysis: a problem has emerged in the network and the data is searched for relevant information to resolve the issue. What is being proposed here is a way to proactively approach the data for situational awareness and potentially uncovering problems in the network that current heuristics and approaches have not discovered. The large volume of these logs necessitates run time complexity less than $O(n^2)$. With these things in mind, the categorization done with this approach will be focused on unsupervised clustering paradigms.

The second consideration is one of hyperparameters. In most clustering approaches, the number of clusters, $k$, is decided *a priori*. Often to find a suitable $k$, a parameter search is performed that evaluates a number of different $k$ values and optimizes a set of criteria, (such as minimized intra-cluster distance/maximized inter-cluster distance) to select an appropriate $k$. The alternative to manually selecting $k$ is to use approaches like hierarchical clustering but these approaches bring with them $O(n^2)$ complexity.

These two considerations drive many of the design decisions of the SRG approach.

### Use Case
SRG is agnostic to log type. It can be trained over a single log source or multiple log sources in a single data set. Models can be trained and fit over the same set to provide immediate insight into a given set or alternatively a model can be trained and saved to categorize an ongoing flow of log messages.

### Technique Overview
The breadth of literature on string resemblance provides a good starting point to solve the problem at hand. So the primary focuses of the problem become the time complexity and hyperparameter selection as discussed in the [Problem Background](#problem-background). This means that the approach explored tries to balance time complexity with data driven hyperparameter selection. For a large number of clustering algorithms, the number of clusters, $k$, must be chosen *a priori*. A search can be performed for an optimal $k$, but this can drastically increase the time complexity.

In order to keep the time complexity low when selecting the number of clusters, SRG works by trying to subset the logs based on different parameters. The number of resulting disjoint subsets becomes the number of representatives, $k$. The first such parameter is the length of each string. When dealing with a varied collection of networking logs, it is often the case that similar logs are similarly sized. Explicitly, the first step of SRG is to construct a kernel density estimate (KDE) of the lengths of the strings to be grouped. The local minima are then found over the range of string lengths and used to perform an initial grouping of the data, which for future reference will be called length groups.

Next the strings are shingled by either $n$-grams or word tokens and collected into a bag-of-words. For each length group, these collections are projected into 1-dimensional spaces using an MDS technique called [FastMap](https://dl.acm.org/doi/abs/10.1145/568271.223812). Once the strings are projected into these 1-dimensional spaces, the KDE's are constructed and the strings are placed into sub-groups based on the local minima of the kernel density estimates. Within these sub-groups, the strings closest to the group maxima are set aside as group representatives.

The last piece of subsetting can be applied using domain knowledge about the specific network logs being analyzed, such as pre-grouping HTTP URL's based on the returned status code. Metadata associated with the logs are typically correlated with the log and logs with similar metadata can be more similar to each other than to strings with different metadata. This can provide more focused situational awareness and better clustering when domain knowledge can be leveraged for the network logs.

A benefit to this approach is that instead of fixing the number of groups, $k$, *a priori*, which is typically needed for clustering algorithms; the kernel density estimate of the 1-dimensional projections dictate the number of clusters. This is still a hyperparameter that needs to be set but there are data driven options that can be applied by using a rule-of-thumb for the bandwidth of the kernel density estimate based on the standard deviation of the data. There are also data driven approaches that can be used to determine an optimal number of clusters to choose, but these are often $O(n^2)$ whereas FastMap is $O(n)$ with approximation techniques for [KDE](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1615&context=statistics_papers) translates into a faster, data driven $k$ selection.

### Model Overview
The model stores the representatives and the means for assigning new messages to a representative group.

### Training Input
A collection of logs. These can be in a text file to be loaded into a Rapids cudf or an already loaded collection.

### Inference Input
A single log instance or collection of logs or text files containing logs.

### Inference Output
The representative and a corresponding numeric label for the log instance or a dataframe containing the original logs and the assigned representative and numeric group.

### API Example
See this [notebook](/string-resemblance-grouping/training-tuning/string-resemblance-grouping.ipynb) for an example on building and inferencing a SRG model.

### Future work
Currently SRG representatives and groups are output as the final result. Future work will instead leverage these representatives as initial "centroids" in a $k$-means variant using the weighted Jaccard distance and iteratively honing centroids that are the mean weights of the present shingles in the group. Once the $k$-means centroids converge or a fixed number of iterations is completed, the closest group member to each centroid is chosen as the final group representative.

Further work will look into bootstrapping the length and FastMap 1-D grouping using ensemble clustering, which is an approach that finds a metacluster label for data that has multiple clustering labels assigned to each data point. The benefit to this, especially in the FastMap grouping, is to smooth out the variance in the 1-D grouping. The impact to runtime is offset by the fact that all of the 1-D groupings can be run simultaneously. This means that the largest impact to the added runtime is just from the ensemble clustering algorithm.

### References
* https://dl.acm.org/doi/abs/10.1145/568271.223812
* https://repository.upenn.edu/cgi/viewcontent.cgi?article=1615&context=statistics_papers

# Model Card ++
# Model Overview

## Description:
String Resemblance Grouping (SRG) is designed to find a subset of representative strings within a a large collection of messages. These representative strings create groupings with which to categorize the messages for further exploration or triage. This particular model was built using Windows log data. <br>

## References(s):
* https://dl.acm.org/doi/abs/10.1145/568271.223812
* https://repository.upenn.edu/cgi/viewcontent.cgi?article=1615&context=statistics_papers  <br> 

## Model Architecture: 
**Architecture Type:** 
* Not Applicable (N/A)  <br>

**Network Architecture:**
* None <br>

## Input: (Enter "None" As Needed)
**Input Format:** 
* String <br>

**Input Parameters:** 
* None <br>

**Other Properties Related to Output:** 
* None <br>

## Output: (Enter "None" As Needed)
**Output Format:** 
* Cluster label and cluster representative <br>

**Output Parameters:** 
* None <br>

**Other Properties Related to Output:** 
* None <br> 

## Software Integration:
**Runtime(s):** 
* Not Applicable (N/A) <br>

**Supported Hardware Platform(s):** <br>
* All <br>

**Supported Operating System(s):** <br>
* Linux <br>

## Model Version(s): 
* 20230627  <br>

# Training & Evaluation: 

## Training Dataset:

**Link:** 
* https://zenodo.org/record/3227177/files/Windows.tar.gz <br>

**Properties (Quantity, Dataset Descriptions, Sensor(s)):** 
* A collection of 114535 Windows logs <br>

**Dataset License:**  
* Owned and hosted by Zenodo <br>

## Evaluation Dataset:
**Link:** 
* https://zenodo.org/record/3227177/files/Windows.tar.gz <br>

**Properties (Quantity, Dataset Descriptions, Sensor(s)):** 
* A collection of 114535 Windows logs <br>

**Dataset License:** 
* Owned and hosted by Zenodo <br>

## Inference:
**Engine:** 
* Other (Not Listed) <br>

**Test Hardware:** <br>
* Other (Not Listed)  <br>

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
* This model is intended to be used to syntactically cluster Windows logs.

### Fill in the blank for the model technique.
* This model is intended for developers that want to build and/or customize syntactic clusters or groupings of a collection of logs.

### Name who is intended to benefit from this model. 
* This model is intended for anyone that wants to syntactically cluster Windows logs for data insight or triage.


### Describe the model output.
* This model outputs a cluster label and the corresponding cluster representative.

### List the steps explaining how this model works.
---
The breadth of literature on string resemblance provides a good starting point to solve the problem at hand. So the primary focuses of the problem become the time complexity and hyperparameter selection as discussed in the [Problem Background](#problem-background). This means that the approach explored tries to balance time complexity with data driven hyperparameter selection. For a large number of clustering algorithms, the number of clusters, $k$, must be chosen *a priori*. A search can be performed for an optimal $k$, but this can drastically increase the time complexity.

In order to keep the time complexity low when selecting the number of clusters, SRG works by trying to subset the logs based on different parameters. The number of resulting disjoint subsets becomes the number of representatives, $k$. The first such parameter is the length of each string. When dealing with a varied collection of networking logs, it is often the case that similar logs are similarly sized. Explicitly, the first step of SRG is to construct a kernel density estimate (KDE) of the lengths of the strings to be grouped. The local minima are then found over the range of string lengths and used to perform an initial grouping of the data, which for future reference will be called length groups.

Next the strings are shingled by either $n$-grams or word tokens and collected into a bag-of-words. For each length group, these collections are projected into 1-dimensional spaces using an MDS technique called [FastMap](https://dl.acm.org/doi/abs/10.1145/568271.223812). Once the strings are projected into these 1-dimensional spaces, the KDE's are constructed and the strings are placed into sub-groups based on the local minima of the kernel density estimates. Within these sub-groups, the strings closest to the group maxima are set aside as group representatives.

The last piece of subsetting can be applied using domain knowledge about the specific network logs being analyzed, such as pre-grouping HTTP URL's based on the returned status code. Metadata associated with the logs are typically correlated with the log and logs with similar metadata can be more similar to each other than to strings with different metadata. This can provide more focused situational awareness and better clustering when domain knowledge can be leveraged for the network logs.

A benefit to this approach is that instead of fixing the number of groups, $k$, *a priori*, which is typically needed for clustering algorithms; the kernel density estimate of the 1-dimensional projections dictate the number of clusters. This is still a hyperparameter that needs to be set but there are data driven options that can be applied by using a rule-of-thumb for the bandwidth of the kernel density estimate based on the standard deviation of the data. There are also data driven approaches that can be used to determine an optimal number of clusters to choose, but these are often $O(n^2)$ whereas FastMap is $O(n)$ with approximation techniques for [KDE](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1615&context=statistics_papers) translates into a faster, data driven $k$ selection.

---

### Name the adversely impacted groups (protected classes) this has been tested to deliver comparable outcomes regardless of:
* Not Applicable

### List the technical limitations of the model.
* Windows log files that are too syntactically different from the training data or from different versions of Windows from the training set.

### What performance metrics were used to affirm the model's performance?
* Cluster spread (mean and standard deviation)

### What are the potential known risks to users and stakeholders?
* None

### What training is recommended for developers working with this model?  If none, please state "none."
* Familiarty with clustering techniques

### Link the relevant end user license agreement 
* [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0) <br>


## Model Card ++ Saftey & Security Subcard

### Link the location of the training dataset's repository (if able to share).
* https://zenodo.org/record/3227177/files/Windows.tar.gz

### Is the model used in an application with physical safety impact?
* No

### Describe physical safety impact (if present).
* N/A

### Was model and dataset assessed for vulnerability for potential form of attack?
* No

### Name applications for the model.
* This model is intended to be used to syntactically cluster Windows logs for data insight and triage.

### Name use case restrictions for the model.
* The model can only be used with Windows log data.

### Has this been verified to have met prescribed quality standards?
* No

### Name target quality Key Performance Indicators (KPIs) for which this has been tested.  
* None

### Technical robustness and model security validated?
* No


### Is the model and dataset compliant with National Classification Management Society (NCMS)?
* Yes

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
* None of the Above


### How often is dataset reviewed?
* Annually

### Is a mechanism in place to honor data subject right of access or deletion of personal data?
* Yes

### If PII collected for the development of this AI model, was it minimized to only what was required? 
* N/A

### Is data in dataset traceable?
* No

### Scanned for malware?
* Yes

### Are we able to identify and trace source of dataset?
* Yes

### Does data labeling (annotation, metadata) comply with privacy laws?
* Yes

### Is data compliant with data subject requests for data correction or removal, if such a request was made?
* No