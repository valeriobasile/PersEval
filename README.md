# PersEval

## A Library for Perspectivist Classification Evaluation

**PersEval** is the first library for the evaluation of perspectivist classification models, developed to streamline the current fragmented perspectivist evaluation practices with a new unified framework. 

The framework is presented in paper [PersEval: A Framework for Perspectivist Classification Evaluation](https://aclanthology.org/2025.emnlp-main.1137), accepted at the main trak of **EMNLP 2025**.

## 📝 Abstract
Data perspectivism goes beyond majority vote label aggregation by recognizing various perspectives as legitimate ground truths.
However, current evaluation practices remain fragmented, making it difficult to compare perspectivist approaches and analyze their impact on different users and demographic subgroups. To address this gap, we introduce PersEval, the first unified framework for evaluating perspectivist models in NLP. A key innovation is its evaluation at the individual annotator level and its treatment of annotators and users as distinct entities, consistently with real-world scenarios.
We demonstrate PersEval's capabilities through experiments with both Encoder-based and Decoder-based approaches, as well as an analysis of the effect of sociodemographic prompting. 
By considering global, text-, trait- and user-level evaluation metrics, we show that PersEval is a powerful tool for examining how models are influenced by user-specific information and identifying the biases this information may introduce. 



## ⚙️ Framework

[ADD IMAGE HERE ]

### Named or unnamed classification
Users can be represented in two ways:

- **Unnamed classification** — by their identifier only.

- **Named classification** — by a set of explicit metadata (e.g., traits).

To enable named classification, use the --named flag when running the script:
```
python main.py --named
```
If the flag is omitted, the script defaults to unnamed classification.


### Adaptation set 
Mirroring real-world scenarios, we assume that the annotators who provided the bulk of the annotations to train the system, and the users for which the system is tested, are disjoint. 

When explicit knowledge about the users is available (e.g. metadata), a model can attempt to learn biases toward such characteristics, without knowing any preference of the unkown users.
 
```
python main.py --named --adaptation false
```

However, we also assume two adaptation scenarios where few labels from test users are available: 

**Adaptation at training time**: we assume minimal annotation from users has been obtained before training the system. A few annotations from test users are thus included in the training split.
```
python main.py --adaptation train
```
```
python main.py --named --adaptation train  
```

**Adaptation at inference time**: we assume an already trained system has to be adapted to new users. A few test users instances can thus be used to adapt an existing model.
```
python main.py --adaptation test
```
```
python main.py --named --adaptation test 
```

### Training set 
In the default setting, textual examples in the test split are disjoint from those in the training split. 


We also provide a variant for which texts that are also found in test instances, but annotated by different users, can be included in the trainig split.
```
python main.py --extended
```

### Test set
The difference between the task variants manifests in different training splits (or in some cases additional sets when using an adaptation set at inference time), while **the test set depends uniquely on the chosen dataset and it remains the same across all task variants.**


## 🗃️ Datasets

The library comes with five datasets. 



| Dataset  | Reference | Task| #Annotators | Metadata | 
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Brexit  | [Akhtar et al. 2021](https://arxiv.org/abs/2106.15896)  | Abusive language | 6 | Target an control group | 
| EPIC  | [Frenda et al. 2023](https://aclanthology.org/2023.acl-long.774/)  | Irony | 74 | Gender, Nationality, Age/Generation | 
| MHS | [Sachdeva et al. 2022](https://aclanthology.org/2022.nlperspectives-1.11/)  | Hate Speech | 7,912 | Gender, Age/Generation, Education, Income, Ideology | 
| MD-Agreement |[Leonaredlli et al. 2022](https://aclanthology.org/2023.semeval-1.314/)  | Offensiveness | 819 | --- | 
|Dices | [Aroyo et al. 2024](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a74b697bce4cac6c91896372abaa8863-Abstract-Datasets_and_Benchmarks.html)  | AI Safety | 123 | Gender, Age/Generation, Education, Ethinicity | 

BREXIT, EPIC and MHS are directly loaded from huggingface, while DICES and MD-Agreement are loaded form the "./data" folder.

Other than the dataset name, is important to also specify the corresponding label.

```
python main.py --dataset-name DICES --label Q2_harmful_content_overall
```
Options for labels are listed above, as well as in the main.py file
```
# EPIC   -> ["irony"]
# BREXIT -> ["hs", "offensiveness", "aggressiveness", "stereotype"]
# DICES  -> ["Q2_harmful_content_overall"]
# MHS    -> ["hateful"]
# MD     -> ["offensiveness"]
```

## 📈 Metrics

We require models to output a **label** for each **<user, text> tuple**. Starting from these prediction, we compute standard classification metrics. Specifically precision, recall and F1 scores for each class, as well as their macro- and micro-average. This same metrics are computed also at more fine-grained levels: 
- annotator-level: computed individually for each annotator and then averaged; 
- text-level: computed individually for each text in the test set and then averaged; 
- trait-level: computed for each trait and then averaged for each dimension.


## 🤖 Baseline Models 
### Encoder-based 
We fine-tuned RoBERTa, customized implementing Focal Loss.

We added annotators' identifiers and their traits to the text embedding as a special token. The model input thus concatenates the annotator id, a special token for each of the annotator's traits, and the input text to classify. 
The model is then trained with a classification head to predict the binary label.

```
python main.py --model-name roberta-base --type encoder
```

### Decoder-based 
We chose on open-source models of mediud size: Mixtral-8 /B and Llama-3.1 8B, both instruction tuned. We considered three possible settings: 

**Base zero**: we prompt the models to classify the test set examples, with no additional information.

```
python main.py --model-name meta-llama/Meta-Llama-3.1-8B-Instruct --type llm
```

**Perspective**: weask the models to impersonate each user’s trait. We use this variant to test models without adaptation with a named user representation. We prompt the model for each available user trait.

```
python main.py --named --model-name meta-llama/Meta-Llama-3.1-8B-Instruct --type llm
```

**In-Prompt Augmentation** We reproduced
 [Salemi et al. (2024)](https://aclanthology.org/2024.acl-long.399/)’s approach,  prompting the model with user-specific input selected via retrieval augmentation. We used this approach both giving information about the user’s trait value (Named with Adaptation-T) and without providing demographic information. 

```
pyton main.py --model-name meta-llama/Meta-Llama-3.1-8B-Instruct --type LaMP --context
```

## 🧰 Our setup
The required dependencies were verified on the following system:

- Operating System: Ubuntu 22.04.3 LTS (Jammy)
- Kernel: Linux 5.15.0-113-generic x86_64
- CPU: AMD EPYC-Rome Processor — 24 cores, 1 thread per core
- RAM: 503 GiB
- GPU: 4× NVIDIA A40 (46 GiB each)
- Python: 3.10.12

Performance and compatibility may vary on different hardware or OS versions.

## 📊 Qualitative analysis
Representing users through their sociodemographic traits could lead to the risk of stereotype propagation. So we evaluated the effect sociodemographic prompting through a qualitative analysis based on the following Questions: 
- Q1: What is the contribution of each trait when ensembling the model’s outputs?
- Q2: Which demographic trait most significantly impacts the model’s label predictions in the presence of varying annotator characteristics?
- Q3: How similar is the distribution of models’ predictions to that of the annotators’ chosen labels

Results are placed at the "./qualitative_analysis" folder.

## Other information

Note that the current version of the library does not support multi-GPU settings.

The In-Prompt Auugmentation strategy is part of a comprehensive evaluation framework for personalization with LLMs (LaMP). The full framework is available at [this repository](https://github.com/LaMP-Benchmark/LaMP/tree/main).