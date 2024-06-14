## Cyber Foundation Model Data

### Cyber Foundation Raw Azure AD Logs
This is a synthetic dataset of Azure AD logs with activities of 20 accounts (85 applications involved, 3567 records in total). The activities are split to a train and an inference set. An anomaly is included in the inference set for model validation. The data was generated using the python [faker](https://faker.readthedocs.io/en/master/#) package. If there is any resemblance to real individuals, it is purely coincidental.

#### Sample Training Data
- 3239 records in total
- Time range: 2022/08/01 - 2022/08/29
- Users' log distribution:
    - 5 high volume (>= 300) users
    - 15 medium volume (~100) users
    - 5 light volume (~10) users

- [./azure-ad-logs-sample-training-data.json](./azure-ad-logs-sample-training-data.json)
- [Original location of the training data](https://github.com/nv-morpheus/Morpheus/blob/main/models/datasets/training-data/azure/azure-ad-logs-sample-training-data.json)