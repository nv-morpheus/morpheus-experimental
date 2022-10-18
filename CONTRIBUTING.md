# Contributing to Morpheus Experimental

Contributions to Morpheus Experimental fall into the following three categories.

## 1. Feature requests and bug reporting

1. To report a bug, request a new feature, or report a problem with documentation, please file an [issue](https://github.com/nv-morpheus/morpheus-experimental/issues/new) describing in detail the problem or new feature. 
2. The Morpheus team evaluates and triages issues If you believe the issue needs priority attention, please comment on the issue to notify the team.


## 2. Full prototype contributions

1. To proprose a new prototype for a cybersecurity workflow, please file an [issue](https://github.com/nv-morpheus/morpheus-experimental/issues/new) describing in detail the prototype including the specific cybersecurity use case it targets, the model architecture, types of input data required (pcap, sflow, application logs, etc.), and any relevant references.
2. The morpheus team and community will discuss the design and impact of the proposed prototype. Once the team agrees that the plan looks good, go ahead and implement it.
3. A prototype should include at minimum a tutorial-style notebook for training and inferencing using the model, model file(s), sample data, training script, inference script, documentation, and a `requirements.txt` file containing all necessary packages to run the scripts. A working morpheus pipeline for the prototype is optional.
4. Make sure the prototype contains subdirectories, its own README, and follows the [repo structure](README.md).
5. When done, [create your pull request](https://github.com/nv-morpheus/morpheus-experimental/compare)
6. Wait for other developers to review your code and update code as needed.
7. Once reviewed and approved, a Morpheus developer will merge your pull request.


## 3. New feature or bug-fix contributions

1. Find an issue to work on. The best way is to look for issues with the [good first issue](https://github.com/nv-morpheus/morpheus-experimental/issues) label.
2. Comment on the issue stating that you are going to work on it.
3. Code! Ensure the [license headers are set properly](#Licensing).
4. When done, [create your pull request](https://github.com/nv-morpheus/morpheus-experimental/compare).
5. Wait for other developers to review your code and update code as needed.
6. Once reviewed and approved, a Morpheus developer will merge your pull request.

Remember, if you are unsure about anything, don't hesitate to comment on issues and ask for clarifications!

As contributors and maintainers to this project,
you are expected to abide by Morpheus Experimental's code of conduct.
More information can be found at: [Contributor Code of Conduct](CODE_OF_CONDUCT.md).

## Licensing
Morpheus is licensed under the Apache v2.0 license. All new source files including CMake and other build scripts should contain the Apache v2.0 license header. Any edits to existing source code should update the date range of the copyright to the current year. The format for the license header is:

```
/*
 * SPDX-FileCopyrightText: Copyright (c) <year>, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 ```

### Thirdparty code
Thirdparty code included in the source tree (that is not pulled in as an external dependency) must be compatible with the Apache v2.0 license and should retain the original license along with a url to the source. If this code is modified, it should contain both the Apache v2.0 license followed by the original license of the code and the url to the original code.

Ex:
```
/**
 * SPDX-FileCopyrightText: Copyright (c) 2018-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//
// Original Source: https://github.com/org/other_project
//
// Original License:
// ...
```


---

## Attribution
Portions adopted from https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md \
Portions adopted from https://github.com/dask/dask/blob/master/docs/source/develop.rst


### Testing CI
don't merge this