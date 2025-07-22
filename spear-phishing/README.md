# Protective Layers Against Targeted E-Mail (PlateMail)

See overview and example results [here](https://developer.nvidia.com/blog/generative-ai-and-accelerated-computing-for-spear-phishing-detection/).

## Exploring the example

[Example notebook](/spear-phishing/trainin-tuning/spear_phishing.ipynb)

### Intent models
The example leverages pre-trained distilBERT intent models, i.e. models trained to detect specific intents within the text. The intents trained for the example are "asking for money", "asking for personally identifiable information (PII)", and "talking about crypto" (for crypto based scams). These intent models were trained using generative LLM techniques to allow for more robust and targeted models. These intent model classifiers then become features for the final spear phishing label for the email. 

### Historical sketching
The example also creates synthetic historic sender data in order to show how we try to catch spoofed spear phishing emails by building historical sender sketches that look at syntactic, temporal, and intention patterns for given senders.

### Running through the example
So the example shows how the workflow can be extensible as novel attacks are discovered. It begins by just using the "money" and "PII" intent classifiers for spear phishing detection. Crypto based attacks are then introduced and classified, where we see that these emails largely escape detection. The "crypto" intent is then added to the spear phishing feature list and training is updated. We then see that these new crpyto attacks are then detected.