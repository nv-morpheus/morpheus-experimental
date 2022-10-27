# Syslog Resemblance Grouping - SRG
SRG is designed to find a subset of representative strings within a a large collection of messages. These representative strings create groupings with which to categorize the messages for further exploration or triage.

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

### Future work
Currently SRG representatives and groups are output as the final result. Future work will instead leverage these representatives as initial "centroids" in a $k$-means variant using the weighted Jaccard distance and iteratively honing centroids that are the mean weights of the present shingles in the group. Once the $k$-means centroids converge or a fixed number of iterations is completed, the closest group member to each centroid is chosen as the final group representative.

Further work will look into bootstrapping the length and FastMap 1-D grouping using ensemble clustering, which is an approach that finds a metacluster label for data that has multiple clustering labels assigned to each data point. The benefit to this, especially in the FastMap grouping, is to smooth out the variance in the 1-D grouping. The impact to runtime is offset by the fact that all of the 1-D groupings can be run simultaneously. This means that the largest impact to the added runtime is just from the ensemble clustering algorithm.

### References
* https://dl.acm.org/doi/abs/10.1145/568271.223812
* https://repository.upenn.edu/cgi/viewcontent.cgi?article=1615&context=statistics_papers