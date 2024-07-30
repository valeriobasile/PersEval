# PersEval

## A Library for Perspectivist Classification Evaluation

**PersEval** is the first library for the evaluation of perspectivist classification models.

We aim to streamline the current fragmented perspectivist evaluation practices with a new unified framework. In this framework, the input consists of a **text** and a **user** who has expressed their perception of a given linguistic phenomenon, encoded by a **label**. 

Users can be represented by their identifier only (**unnamed classification**) or by a set of explicit metadata, i.e., traits (**named classification**). 

Mirroring real-world scenarios, we assume that the annotators who provided the bulk of the annotations to train the system, and the users for which the system is tested, are disjoint; we also consider some relaxations of this hypothesis, assuming few labels from test users are known either at training time or after a first training is completed to adapt the trained system. 

Given a set of predictions, we provide standard global metrics and fine-grained metrics at the user, text, and trait levels.

We provide intuitive functions to access the data and the evaluation metrics.

We showcase its use in evaluating a baseline model on two diverse disaggregated datasets.

You can find an example of usage by running `python3 main.py`. 

Note that the current version of the library does not support multi-GPU settings.

