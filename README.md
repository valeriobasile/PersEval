# PersEval

## A Library for Perspectivist Classification Evaluation

**PersEval** is the first library for the evaluation of perspectivist classification models, developed to streamline the current fragmented perspectivist evaluation practices with a new unified framework. 
For more information about Perspectivism, see the manifesto page: https://pdai.info/

In the proposed framework, the input consists of a **text** and a **user** who has expressed their perception of a given linguistic phenomenon, encoded by a **label**. 

Given a set of predictions, we provide standard global metrics and fine-grained metrics at the user, text, and trait levels.

We provide intuitive functions to access the data and the evaluation metrics.

Finally, we showcase its use in evaluating a baseline model on two diverse disaggregated datasets.

You can find an example of usage by running `python3 main.py`. 


## Framework

Users can be represented by their identifier only (**unnamed classification**) or by a set of explicit metadata, i.e., traits (**named classification**). 

Mirroring real-world scenarios, we assume that the annotators who provided the bulk of the annotations to train the system, and the users for which the system is tested, are disjoint; we also consider some relaxations of this hypothesis, assuming few labels from test users are known either at training time or after a first training is completed to adapt the trained system.

In the default setting, textual examples in the test split are disjoint from those in the training split. We also provide a variant for which texts that are also found in test instances, but annotated by different users, can be included in the trainig split.

The difference between the task variants manifests in different training splits (or in some cases additional sets when using an adaptation set at inference time), while the test set depends uniquely on the chosen dataset and it remains the same across all task variants.


## Datasets

The library comes with two datasets directly loaded from HuggingFace: 
- BREXIT, dense dataset where each instance is annotated by all annotators (https://huggingface.co/datasets/silvia-casola/BREXIT)
- EPIC, crowdsourced and sparse dataset (https://huggingface.co/datasets/Multilingual-Perspectivist-NLU/EPICbre)


## Metrics

We require models to output a **label** for each **<user, text> tuple**. Starting from these prediction, we compute standard classification metrics. Specifically precision, recall and F1 scores for each class, as well as their macro- and micro-average. This same metrics are computed also at more fine-grained levels: 
- annotator-level: computed individually for each annotator and then averaged; 
- text-level: computed individually for each text in the test set and then averaged; 
- trait-level: computed for each trait and then averaged for each dimension.


## Setup

`data` submodule allows to instantiate a dataset. 

The user can then request the training, test, and optionally adaptation data splits with the `get\_splits() method`, indicating whether the adaptation data (`user\_adaptation`) is absent (`False`), available at training time (`train`) or at inference time (`test`). 

Additionally, the user chooses whether to extend the training split including texts also in test instances (`extended=True`) or to exclude them (`extended=False`). 

## Baseline Models 
We fine-tuned RoBERTa, customized implementing Focal Loss.

We added annotators' identifiers and their traits to the text embedding as a special token. The model input thus concatenates the annotator id, a special token for each of the annotator's traits, and the input text to classify. 
The model is then trained with a classification head to predict the binary label.

## Other information

Note that the current version of the library does not support multi-GPU settings.
