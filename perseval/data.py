import logging as log
from random import seed, sample
from dataclasses import dataclass  
import copy                                             

from tqdm import tqdm
import numpy as np
from datasets import load_dataset, concatenate_datasets, load_from_disk
from sklearn.model_selection import train_test_split

from . import config

log.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    encoding='utf-8', 
    level=log.INFO)

# Changing the random seed will change how the datasets are split
seed(config.seed)

@dataclass
class PerspectivistDataset:
    def __init__(self):
        self.name = None
        self.traits = {}
        self.labels = dict()
        self.training_set = None
        self.adaptation_set = None
        self.test_set = None
        self.user_adaptation = None
        self.named = None
        self.extended = None

    def describe_splits(self):
        if not self.training_set.users:
            raise Exception("You need to first choose a task through "+self.name+".get_splits(extended, user_adaptation, named,baseline)")
        
        print("--- Unique users ---")
        print("Train set: %d" % len(self.training_set.users))
        if len(self.adaptation_set.users):
            print("Adaptation set: %d" % len(self.adaptation_set.users))
        print("Test set: %d" % len(self.test_set.users))
        print()
        print("--- Unique texts ---")
        print("Train set: %d" % len(self.training_set.annotation_by_text))
        if len(self.adaptation_set.annotation_by_text):
            print("Adaptation set: %d" % len(self.adaptation_set.annotation_by_text))
        print("Test set: %d" % len(self.test_set.annotation_by_text))
        print()
        print("--- Instances (text + user) ---")
        print("Train set: %d" % len(self.training_set.annotation))
        if len(self.adaptation_set.annotation):
            print("Adaptation set: %d" % len(self.adaptation_set.annotation))
        print("Test set: %d" % len(self.test_set.annotation))
        print()

        print("--- User-text train/adaptation/test distribution ---")
        number_user_adapt_texts, number_user_test_texts = [], []
        for u in self.test_set.users:
            user_adapt_texts, user_test_texts = 0, 0
            for i in self.adaptation_set.annotation:
                if i[0]==u:
                    user_adapt_texts+=1
            for i in self.test_set.annotation:
                if i[0]==u:
                    user_test_texts+=1
            number_user_adapt_texts.append(user_adapt_texts)
            number_user_test_texts.append(user_test_texts)
        print("The mean number of texts per users in the test set is %.3f" % np.mean(number_user_test_texts))
        
        if self.adaptation_set != PerspectivistSplit(type=="adaptation"):
            percentage_in_adapt = [d/t for d, t in zip(number_user_adapt_texts, number_user_test_texts)]
            print("The mean percentage of texts per users in the adaptation set is %.3f" % np.mean(percentage_in_adapt))
            print("The mean number of texts per users in the adaptation set is %.3f" % np.mean(number_user_adapt_texts))
            print("The min percentage of texts per users in the adaptation set is %.3f (i.e. %.0f instances)" % (np.min(percentage_in_adapt), np.min(number_user_adapt_texts)))
            print("The max percentage of texts per users in the adaptation set is %.3f (i.e. %.0f instances)" % (np.max(percentage_in_adapt), np.max(number_user_adapt_texts)))


    def check_splits(self, user_adaptation, extended, named):
        if user_adaptation == False:
            # The adaptation set is empty
            assert self.adaptation_set == PerspectivistSplit(type="adaptation")
        
        # Users
        if user_adaptation == False or user_adaptation == "test":
            # Train and adapt + test users have no overlap
            assert set(self.training_set.users).intersection(set(self.adaptation_set.users)) == set()
            assert set(self.training_set.users).intersection(set(self.test_set.users)) == set()
        if user_adaptation == "train":
            # All test users are also in the training set
            assert set(self.training_set.users).union(set(self.test_set.users)) == set(self.training_set.users) 
        
        # Texts
        # adapt and test texts have no overlap
        if user_adaptation == "test":
            assert set(self.adaptation_set.texts).intersection(set(self.test_set.texts)) == set()  

        for u in self.test_set.users:
            user_train_texts, user_adapt_texts, user_test_texts = 0, 0, 0
            for i in self.training_set.annotation:
                if i[0]==u:
                    user_train_texts+=1 
            for i in self.adaptation_set.annotation:
                if i[0]==u:
                    user_adapt_texts+=1
            for i in self.test_set.annotation:
                if i[0]==u:
                    user_test_texts+=1
            
        if user_adaptation == "train" and extended:
            # All test users and corresponding training users must have at least one annotation
            assert user_train_texts != 0
            assert user_test_texts != 0
        if user_adaptation == "test":
            # All test users and corresponding adapt users must have at least one annotation
            assert user_adapt_texts != 0
            assert user_test_texts != 0

        if not extended:
            # Train and test text have no overlap
            assert set(self.training_set.texts).intersection(set(self.test_set.texts)) == set()  
        log.info("All tests passed")


@dataclass
class Instance:
    def __init__(self, instance_id, instance_text, user, label):
        self.instance_id = instance_id
        self.instance_text = instance_text
        self.user = user
        self.label = label

    def __repr__(self):
        return f"{self.instance_id} {self.user} {self.label}"


@dataclass
class PerspectivistSplit:
    def __init__(self, type=None):
        self.type = type # Str, e.g., train, adaptation, test
        self.users = dict() 
        self.texts = dict()
        self.annotation = dict() #user, text, label
        self.annotation_by_text = dict()

    def __iter__(self):
        for (user, instance_id), label in self.annotation.items():
            yield Instance(
                instance_id, 
                self.texts[instance_id],
                self.users[user],
                label)

    def __len__(self):
        return len(self.annotation)
    

@dataclass
class User:
    def __init__(self, user):
        self.id = user
        self.traits = dict()

    def __lt__(self, other):
        return self.id < other.id
    
    def __eq__(self, other):
        if self.id == other.id and self.traits == other.traits:
            return True
        else:
            return False


@dataclass
class Epic(PerspectivistDataset):
    def __init__(self):
        super(Epic, self).__init__()
        self.name = "EPIC"
        dataset = load_dataset("Multilingual-Perspectivist-NLU/EPIC")
        self.dataset = dataset["train"]
        self.dataset = self.dataset.map(lambda x: {"label": {"iro":1, "not":0}[x["label"]]})
        self.label = config.dataset_label[self.name]
        self.labels["irony"] = set()

    def get_splits(self, extended, user_adaptation, named, baseline=False):
        if not user_adaptation in [False, "train", "test"]:
            raise Exception(
                "Possible values are:\n \
                - False (bool): No adaptation is performed. The train and test splits are completly disjoint. The adaptation split is empty.\n \
                - 'train' (str): A small percentage (defined in the config) of the annotations by test users is contained in the training split. The adaptation split is empty. This mirrors a situation in which one can obtain a minimal amount of annotationd *before* training the system.\n \
                - 'test' (str): A small percentage (defined in the config) of the annotations by the test user is in the adapatation split. This mirrors a situation in which one has a trained system (trained on the training users, with no annotations from the test users) and want to adapt the system *after* training it.\n"
                )

        log.info("Generating Named: %s, User adaptation: %s, Extended: %s" % (named, user_adaptation, extended))

        
        self.user_adaptation = user_adaptation
        self.named = named
        self.extended = extended

        self.training_set = self.adaptation_set = self.test_set = None

        if (not user_adaptation and not named) and not baseline:
            raise Exception("Invalid parameter configuration (user_adaptation=False, named=False). \
                            You need to at least know the explicit user traits for test users if no annotations are available")
        
        user_ids = set(list(self.dataset['user']))

        # Sample adapt+test users
        seed(config.seed)
        adaptation_test_user_ids = sample(sorted(user_ids), int(len(user_ids) * config.dataset_specific_splits[self.name]["user_based_split_percentage"]))
        train_user_ids = [u for u in user_ids if not u in adaptation_test_user_ids]
        adapt_test_text_id = [t_id for t_id, user in zip(self.dataset["id_original"], self.dataset["user"]) if user in adaptation_test_user_ids]
        seed(config.seed)
        adaptation_text_ids = sample(sorted(adapt_test_text_id), int(len(adapt_test_text_id) * config.dataset_specific_splits[self.name]["text_based_split_percentage"]))
        test_text_ids = [t_id for t_id in adapt_test_text_id if t_id not in adaptation_text_ids]

        train_split , adaptation_split, test_split = PerspectivistSplit(type="train"), PerspectivistSplit(type="adaptation"), PerspectivistSplit(type="test")
        splits = [train_split, adaptation_split, test_split]
        for split in splits:
            for row in tqdm(self.dataset):
                # Read user
                if (row['user'] in train_user_ids and split.type=="train") or \
                    (row['user'] in adaptation_test_user_ids and split.type=="adaptation") or \
                      (row['user'] in adaptation_test_user_ids and split.type=="test"):
                    if not row['user'] in split.users:
                        split.users[row['user']] = User(row['user'])
                    
                    # Read traits only if named
                    if named:
                        split.users[row['user']].traits["Gender"]=[row['Sex']]
                        if "Gender" in self.traits:
                            self.traits["Gender"].add(row["Sex"])
                        else:
                            self.traits["Gender"] = {(row["Sex"])}

                        split.users[row['user']].traits["Nationality"]=[row['Nationality']]
                        if "Nationality" in self.traits:
                            self.traits["Nationality"].add(row["Nationality"])
                        else:
                            self.traits["Nationality"] = {(row["Nationality"])}
                        try:
                            generation = self.__convert_age(int(row['Age']))
                            split.users[row['user']].traits["Generation"]=[generation]
                            if "Generation" in self.traits:
                                self.traits["Generation"].add(generation)
                            else:
                                self.traits["Generation"] = {generation}
                        except ValueError as e:
                            split.users[row['user']].traits["Generation"]=["UNK"]
                    
                # Read text
                if (row['user'] in train_user_ids and split.type=="train") or \
                    (row['user'] in adaptation_test_user_ids and row['id_original'] in adaptation_text_ids and split.type=="adaptation") or \
                        (row['user'] in adaptation_test_user_ids and row['id_original'] in test_text_ids and split.type=="test"):
                    split.texts[row['id_original']] = {"post": row['parent_text'], "reply": row['text']} 
                
                # Read annotation
                if (row['user'] in train_user_ids and split.type=="train") or \
                    (row['user'] in adaptation_test_user_ids and row['id_original'] in adaptation_text_ids and split.type=="adaptation") or \
                        (row['user'] in adaptation_test_user_ids and row['id_original'] in test_text_ids and split.type=="test"):
                    split.annotation[(row['user'], row['id_original'])] = {}
                    split.annotation[(row['user'], row['id_original'])]["irony"] = row['label']
                    self.labels["irony"].add(row['label'])

                # Read labels by text
                if (row['user'] in train_user_ids and split.type=="train") or \
                    (row['user'] in adaptation_test_user_ids and row['id_original'] in adaptation_text_ids and split.type=="adaptation") or \
                        (row['user'] in adaptation_test_user_ids and row['id_original'] in test_text_ids and split.type=="test"):
                    if not row['id_original'] in split.annotation_by_text:
                        split.annotation_by_text[row['id_original']] = []
                    split.annotation_by_text[row['id_original']].append(
                        {"user": split.users[row['user']], "label": {"irony": row['label']}})
                    self.labels["irony"].add(row['label'])
        
        if user_adaptation == False:
            # You know nothing about the new test users except their explicit traits
            # You cannot use their adaptation annotations
            self.training_set = train_split
            self.adaptation_set = PerspectivistSplit(type="adaptation")
            self.test_set = test_split
                
        elif user_adaptation == "train":
            # You can use a few annotations by test users at training time
            # These annotations are directly included in the training split, 
            # the adaptation split is empty

            # Train + Adapt in the train set
            train_split.users = {**train_split.users, **adaptation_split.users}
            train_split.texts = {**train_split.texts, **adaptation_split.texts}
            train_split.annotation = {**train_split.annotation, **adaptation_split.annotation}

            for t_id in adaptation_split.annotation_by_text.keys():
                if t_id in train_split.annotation_by_text:
                    # add the annotatios
                    train_split.annotation_by_text[t_id] = train_split.annotation_by_text[t_id] + adaptation_split.annotation_by_text[t_id]
                else:
                    train_split.annotation_by_text[t_id] = adaptation_split.annotation_by_text[t_id]
            self.training_set = train_split
            self.adaptation_set = PerspectivistSplit(type="adaptation")
            self.test_set = test_split

                
        elif user_adaptation == "test":
            # You CANNOT use any test annotations at training time
            # However, you can use a few annotations to adapt your trained system to test users 
            # These adaptation annotations from test users are in the adaptation split, 
            self.training_set = train_split
            self.adaptation_set = adaptation_split
            self.test_set = test_split

        if not extended:
            strict_train_split = self.training_set
            strict_train_split.annotation_by_text = {t:self.training_set.annotation_by_text[t] for t in self.training_set.annotation_by_text if t not in self.test_set.annotation_by_text}
            # Filter annotations
            for u, t in copy.deepcopy(self.training_set.annotation):
                if t in self.test_set.annotation_by_text:
                    strict_train_split.annotation.pop((u, t))
    
            # Filter texts
            strict_train_split.texts = {k:self.training_set.texts[k] for k in self.training_set.texts if not k in self.test_set.texts}
            self.training_set = strict_train_split

        self.check_splits(user_adaptation, extended, named)
        self.describe_splits()
        

    def __convert_age(self, age):
        """Function to convert the age, represented as an integer,
        into a label, according to Table 1 in the paper
        'EPIC: Multi-Perspective Annotation of a Corpus of Irony'
        https://aclanthology.org/2023.acl-long.774/
        """
        if age >= 58:
            return "Boomer"
        elif age >= 42:
            return "GenX"
        elif age >= 26:
            return "GenY"
        else:
            return "GenZ"


@dataclass
class Brexit(PerspectivistDataset):
    def __init__(self):
        super(Brexit, self).__init__()
        self.name = "BREXIT"
        dataset = load_dataset("silvia-casola/BREXIT")
        self.dataset = concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])
        labels = ["hs", "offensiveness", "aggressiveness", "stereotype"]
        self.label = config.dataset_label[self.name]
        for label in labels:
            self.labels[label] = set()

    def get_splits(self, extended, user_adaptation, named, baseline=False):
        if not user_adaptation in [False, "train", "test"]:
            raise Exception(
                "Possible values are:\n \
                - False (bool): No adaptation is performed. The train and test splits are completly disjoint. The adaptation split is empty.\n \
                - 'train' (str): A small percentage (defined in the config) of the annotations by test users is contained in the training split. The adaptation split is empty. This mirrors a situation in which one can obtain a minimal amount of annotationd *before* training the system.\n \
                - 'test' (str): A small percentage (defined in the config) of the annotations by the test user is in the adapatation split. This mirrors a situation in which one has a trained system (trained on the training users, with no annotations from the test users) and want to adapt the system *after* training it.\n"
                )
        
        self.user_adaptation = user_adaptation
        self.named = named
        self.extended = extended

        log.info("Generating. Named: %s, User adaptation: %s, Extended: %s" % (named, user_adaptation, extended))
        self.training_set = self.adaptation_set = self.test_set = None

        if (not user_adaptation and not named) and not baseline:
            raise Exception("Invalid parameter configuration (user_adaptation=False, named=False). \
                            You need to at least know the explicit user traits for test users if no annotations are available")
        
        users_group_ids = self.dataset.to_pandas()[["annotator_id", "annotator_group"]].drop_duplicates()
        user_ids = list(users_group_ids['annotator_id'])
        user_group = list(users_group_ids['annotator_group']) 

        # Sample adapt+test users
        seed(config.seed)
        train_user_ids, adaptation_test_user_ids = train_test_split(user_ids, 
                                                        test_size=config.dataset_specific_splits[self.name]["user_based_split_percentage"], 
                                                        random_state=config.seed, 
                                                        shuffle=True, stratify=user_group)        
        seed(config.seed)
        all_text_ids = list(set(self.dataset["instance_id"]))
        train_text_ids = sample(sorted(all_text_ids), int(len(all_text_ids)*config.dataset_specific_splits[self.name]["text_based_split_percentage_train"]))
        adaptation_test_text_ids = [t for t in all_text_ids if t not in train_text_ids]
        adaptation_text_ids = sample(sorted(adaptation_test_text_ids), int(len(adaptation_test_text_ids)*config.dataset_specific_splits[self.name]["text_based_split_percentage_dev"]))
        test_text_ids = [t for t in adaptation_test_text_ids if t not in adaptation_text_ids]

        train_split, adaptation_split, test_split = PerspectivistSplit(type="train"), PerspectivistSplit(type="adaptation"), PerspectivistSplit(type="test")
        splits = [train_split, adaptation_split, test_split]
        for split in splits:
            for row in tqdm(self.dataset):
                # Read user
                if (row['annotator_id'] in train_user_ids and split.type=="train") or \
                    (row['annotator_id'] in adaptation_test_user_ids and split.type=="adaptation") or \
                      (row['annotator_id'] in adaptation_test_user_ids and split.type=="test"):
                    if not row['annotator_id'] in split.users:
                        split.users[row['annotator_id']] = User(row['annotator_id'])
                    
                    # Read traits only if named
                    if named:
                        split.users[row['annotator_id']].traits["Group"]=[row['annotator_group']]                        
                        if "Group" in self.traits:
                            self.traits["Group"].add(row['annotator_group'])
                        else:
                            self.traits["Group"] = {row['annotator_group']}

                # Read text
                if (row['annotator_id'] in train_user_ids and split.type=="train") or \
                    (row['annotator_id'] in adaptation_test_user_ids and  row['instance_id'] in adaptation_text_ids and split.type=="adaptation") or \
                        (row['annotator_id'] in adaptation_test_user_ids and row['instance_id'] in test_text_ids and split.type=="test"):
                   split.texts[row['instance_id']] = {"tweet": row['tweet']}
                
                # Read annotation
                if (row['annotator_id'] in train_user_ids and split.type=="train") or \
                    (row['annotator_id'] in adaptation_test_user_ids and  row['instance_id'] in adaptation_text_ids and split.type=="adaptation") or \
                        (row['annotator_id'] in adaptation_test_user_ids and row['instance_id'] in test_text_ids and split.type=="test"):
                    split.annotation[(row['annotator_id'], row['instance_id'])] = {}
                    for label in self.labels:
                        split.annotation[(row['annotator_id'], row['instance_id'])][label] = row[label]
                        self.labels[label].add(row[label])

                # Read labels by text
                if (row['annotator_id'] in train_user_ids and split.type=="train") or \
                    (row['annotator_id'] in adaptation_test_user_ids and  row['instance_id'] in adaptation_text_ids and split.type=="adaptation") or \
                        (row['annotator_id'] in adaptation_test_user_ids and row['instance_id'] in test_text_ids and split.type=="test"):
                    if not row['instance_id'] in split.annotation_by_text:
                        split.annotation_by_text[row['instance_id']] = []
                    labels_dict = {label: row[label] for label in self.labels}    
                    split.annotation_by_text[row['instance_id']].append(
                        {"user": split.users[row['annotator_id']], "label": labels_dict})
                    for label in self.labels:
                        self.labels[label].add(row[label])
                
        if user_adaptation == False:
            # You know nothing about the new test users except their explicit traits
            # You cannot use their adaptation annotations
            self.training_set = train_split
            self.adaptation_set = PerspectivistSplit(type="adaptation")
            self.test_set = test_split
                
        elif user_adaptation == "train":
            # You can use a few annotations by test users at training time
            # These annotations are directly included in the training split, 
            # the adaptation split is empty

            # Train + Adapt in the train set
            train_split.users = {**train_split.users, **adaptation_split.users}
            train_split.texts = {**train_split.texts, **adaptation_split.texts}
            train_split.annotation = {**train_split.annotation, **adaptation_split.annotation}

            for t_id in adaptation_split.annotation_by_text.keys():
                if t_id in train_split.annotation_by_text:
                    # add the annotatios
                    train_split.annotation_by_text[t_id] = train_split.annotation_by_text[t_id] + adaptation_split.annotation_by_text[t_id]
                else:
                    train_split.annotation_by_text[t_id] = adaptation_split.annotation_by_text[t_id]
            self.training_set = train_split
            self.adaptation_set = PerspectivistSplit(type="adaptation")
            self.test_set = test_split

                
        elif user_adaptation == "test":
            # You CANNOT use any test annotations at training time
            # However, you can use a few annotations to adapt your trained system to test users 
            # These adaptation annotations from test users are in the adaptation split, 
            self.training_set = train_split
            self.adaptation_set = adaptation_split
            self.test_set = test_split
        
        if not extended:
            strict_train_split = self.training_set
            strict_train_split.annotation_by_text = {t:self.training_set.annotation_by_text[t] for t in self.training_set.annotation_by_text if t not in self.test_set.annotation_by_text}
            # Filter annotations
            for u, t in copy.deepcopy(self.training_set.annotation):
                if t in self.test_set.annotation_by_text:
                    strict_train_split.annotation.pop((u, t))
    
            # Filter texts
            strict_train_split.texts = {k:self.training_set.texts[k] for k in self.training_set.texts if not k in self.test_set.texts}
            self.training_set = strict_train_split

        self.check_splits(user_adaptation, extended, named)
        self.describe_splits()
        
@dataclass
class MHS(PerspectivistDataset):
    def __init__(self):
        super(MHS, self).__init__()
        self.name = "MHS"
        dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech")
        self.dataset = dataset["train"]
        self.label = config.dataset_label[self.name]
        self.dataset = self.dataset.map(lambda x: {"hateful": 1 if x["hatespeech"] > 0 else 0})
        self.labels["hateful"] = set()

    def get_splits(self, extended, user_adaptation, named, baseline = False):
        if not user_adaptation in [False, "train", "test"]:
            raise Exception(
                "Possible values are:\n \
                - False (bool): No adaptation is performed. The train and test splits are completly disjoint. The adaptation split is empty.\n \
                - 'train' (str): A small percentage (defined in the config) of the annotations by test users is contained in the training split. The adaptation split is empty. This mirrors a situation in which one can obtain a minimal amount of annotationd *before* training the system.\n \
                - 'test' (str): A small percentage (defined in the config) of the annotations by the test user is in the adapatation split. This mirrors a situation in which one has a trained system (trained on the training users, with no annotations from the test users) and want to adapt the system *after* training it.\n"
                )

        log.info("Generation Named: %s, User adaptation: %s, Extended: %s" % (named, user_adaptation, extended))

        
        self.user_adaptation = user_adaptation
        self.named = named
        self.extended = extended

        self.training_set = self.adaptation_set = self.test_set = None

        if (not user_adaptation and not named) and not baseline:
            raise Exception("Invalid parameter configuration (user_adaptation=False, named=False). \
                            You need to at least know the explicit user traits for test users if no annotations are available")
        
        user_ids = set(list(self.dataset['annotator_id']))

        # Sample adapt+test users
        seed(config.seed)
        adaptation_test_user_ids = sample(sorted(user_ids), int(len(user_ids) * config.dataset_specific_splits[self.name]["user_based_split_percentage"]))
        train_user_ids = [u for u in user_ids if not u in adaptation_test_user_ids]
        adapt_test_text_id = [t_id for t_id, user in zip(self.dataset["comment_id"], self.dataset["annotator_id"]) if user in adaptation_test_user_ids]
        seed(config.seed)
        adaptation_text_ids = sample(sorted(adapt_test_text_id), int(len(adapt_test_text_id) * config.dataset_specific_splits[self.name]["text_based_split_percentage"]))
        test_text_ids = [t_id for t_id in adapt_test_text_id if t_id not in adaptation_text_ids]

        train_split , adaptation_split, test_split = PerspectivistSplit(type="train"), PerspectivistSplit(type="adaptation"), PerspectivistSplit(type="test")
        splits = [train_split, adaptation_split, test_split]
        education={"college_grad_aa":"educ-high","college_grad_ba":"educ-high","high_school_grad":"educ-low","masters":"educ_high","phd":"educ-high","professional_degree":"educ-low","some_college":"educ-low","some_high_school":"educ-low"}
        ideology={"conservative":"conservative","extremely_conservative":"conservative","extremely_liberal":"liberal","liberal":"liberal","neutral":"neutral","no_opinion":"neutral","slightly_conservative":"conservative","slightly_liberal":"liberal"}
        income={"100k-200k":"income-high","10k-50k":"income-low","<10k":"income-low",">200k":"income-high","50k-100k":"income-high"}
        for split in splits:
            for row in tqdm(self.dataset):
                # Read user
                if (row['annotator_id'] in train_user_ids and split.type=="train") or \
                    (row['annotator_id'] in adaptation_test_user_ids and split.type=="adaptation") or \
                    (row['annotator_id'] in adaptation_test_user_ids and split.type=="test"):
                    if not row['annotator_id'] in split.users:
                        split.users[row['annotator_id']] = User(row['annotator_id'])
                    
                    # Read traits only if named
                    if named:
                        
                        # Education
                        if row['annotator_educ'] is not None:
                            split.users[row['annotator_id']].traits["Education"]=[education[row['annotator_educ']]]
                            if "Education" in self.traits:
                                self.traits["Education"].add(education[row["annotator_educ"]])
                            else:
                                self.traits["Education"] = {(education[row["annotator_educ"]])}
                            
                        # Gender
                        split.users[row['annotator_id']].traits["Gender"]=[row['annotator_gender']]
                        if "Gender" in self.traits:
                            self.traits["Gender"].add(row["annotator_gender"])
                        else:
                            self.traits["Gender"] = {(row["annotator_gender"])}
                            
                        # Ideology
                        if row['annotator_ideology'] is not None:
                            split.users[row['annotator_id']].traits["Ideology"]=[ideology[row['annotator_ideology']]]
                            if "Ideology" in self.traits:
                                self.traits["Ideology"].add(ideology[row["annotator_ideology"]])
                            else:
                                self.traits["Ideology"] = {(ideology[row["annotator_ideology"]])}
                            
                        # Race
                        for race in ["asian","black","latinx","middle_eastern","native_american","pacific_islander","white","other"]:
                            split.users[row['annotator_id']].traits["Race-"+race.replace("_","-")]=[row['annotator_race_'+race]]
                            if "Race-"+race.replace("_","-") in self.traits:
                                self.traits["Race-"+race.replace("_","-")].add("yes" if row["annotator_race_"+race] == True else "no")
                            else:
                                self.traits["Race-"+race.replace("_","-")] = {("yes" if row["annotator_race_"+race] == True else "no")}
                            
                        # Income
                        if row['annotator_income'] is not None:
                            split.users[row['annotator_id']].traits["Income"]=[income[row['annotator_income']]]
                            if "Income" in self.traits:
                                self.traits["Income"].add(income[row["annotator_income"]])
                            else:
                                self.traits["Income"] = {(income[row["annotator_income"]])}
                                
                        # Age 
                        if row['annotator_age'] is not None:
                            split.users[row['annotator_id']].traits["Age"]=[self.__convert_age(int(row['annotator_age']))]
                            if "Age" in self.traits:
                                self.traits["Age"].add(self.__convert_age(int(row['annotator_age'])))
                            else:
                                self.traits["Age"] = {(self.__convert_age(int(row['annotator_age'])))}
                    
                # Read text
                if (row['annotator_id'] in train_user_ids and split.type=="train") or \
                    (row['annotator_id'] in adaptation_test_user_ids and row['comment_id'] in adaptation_text_ids and split.type=="adaptation") or \
                        (row['annotator_id'] in adaptation_test_user_ids and row['comment_id'] in test_text_ids and split.type=="test"):
                    split.texts[row['comment_id']] = {"post": row['text']} 
                
                # Read annotation
                if (row['annotator_id'] in train_user_ids and split.type=="train") or \
                    (row['annotator_id'] in adaptation_test_user_ids and row['comment_id'] in adaptation_text_ids and split.type=="adaptation") or \
                        (row['annotator_id'] in adaptation_test_user_ids and row['comment_id'] in test_text_ids and split.type=="test"):
                    split.annotation[(row['annotator_id'], row['comment_id'])] = {}
                    split.annotation[(row['annotator_id'], row['comment_id'])]["hateful"] = 1 if row["hatespeech"] > 0 else 0
                    self.labels["hateful"].add(1 if row["hatespeech"] > 0 else 0)

                # Read labels by text
                if (row['annotator_id'] in train_user_ids and split.type=="train") or \
                    (row['annotator_id'] in adaptation_test_user_ids and row['comment_id'] in adaptation_text_ids and split.type=="adaptation") or \
                        (row['annotator_id'] in adaptation_test_user_ids and row['comment_id'] in test_text_ids and split.type=="test"):
                    if not row['comment_id'] in split.annotation_by_text:
                        split.annotation_by_text[row['comment_id']] = []
                    split.annotation_by_text[row['comment_id']].append(
                        {"user": split.users[row['annotator_id']], "label": {"hateful":1 if row["hatespeech"] > 0 else 0}})
                    self.labels["hateful"].add(1 if row["hatespeech"] > 0 else 0)
        
        if user_adaptation == False:
            # You know nothing about the new test users except their explicit traits
            # You cannot use their adaptation annotations
            self.training_set = train_split
            self.adaptation_set = PerspectivistSplit(type="adaptation")
            self.test_set = test_split
                
        elif user_adaptation == "train":
            # You can use a few annotations by test users at training time
            # These annotations are directly included in the training split, 
            # the adaptation split is empty

            # Train + Adapt in the train set
            train_split.users = {**train_split.users, **adaptation_split.users}
            train_split.texts = {**train_split.texts, **adaptation_split.texts}
            train_split.annotation = {**train_split.annotation, **adaptation_split.annotation}

            for t_id in adaptation_split.annotation_by_text.keys():
                if t_id in train_split.annotation_by_text:
                    # add the annotations
                    train_split.annotation_by_text[t_id] = train_split.annotation_by_text[t_id] + adaptation_split.annotation_by_text[t_id]
                else:
                    train_split.annotation_by_text[t_id] = adaptation_split.annotation_by_text[t_id]
            self.training_set = train_split
            self.adaptation_set = PerspectivistSplit(type="adaptation")
            self.test_set = test_split

                
        elif user_adaptation == "test":
            # You CANNOT use any test annotations at training time
            # However, you can use a few annotations to adapt your trained system to test users 
            # These adaptation annotations from test users are in the adaptation split, 
            self.training_set = train_split
            self.adaptation_set = adaptation_split
            self.test_set = test_split

        if not extended:
            strict_train_split = self.training_set
            strict_train_split.annotation_by_text = {t:self.training_set.annotation_by_text[t] for t in self.training_set.annotation_by_text if t not in self.test_set.annotation_by_text}
            # Filter annotations
            for u, t in copy.deepcopy(self.training_set.annotation):
                if t in self.test_set.annotation_by_text:
                    strict_train_split.annotation.pop((u, t))
    
            # Filter texts
            strict_train_split.texts = {k:self.training_set.texts[k] for k in self.training_set.texts if not k in self.test_set.texts}
            self.training_set = strict_train_split

        self.check_splits(user_adaptation, extended, named)
        self.describe_splits()

    def __convert_age(self, age):
        """Function to convert the age, represented as an integer,
        into a label. 
        The annotations were done in 2020, so the labels are based on 2020.
        """
        if age >= 56:
            return "Boomer"
        elif age >= 40:
            return "GenX"
        elif age >= 24:
            return "GenY"
        else:
            return "GenZ"


@dataclass
class DICES(PerspectivistDataset):
    def __init__(self):
        super(DICES, self).__init__()
        self.name = "DICES-350"
        self.dataset = load_from_disk("data/diverse_safety_adversarial_dialog_350_enhanced")
        self.dataset = self.dataset.map(lambda x: {"degree_of_harm": {"Extreme":3, "Moderate":2, "Debatable":1, "Benign":0}[x["degree_of_harm"]]})
        self.labels["degree_of_harm"] = set()
        self.label = config.dataset_label[self.name]


    def get_splits(self, extended, user_adaptation, named):
        if not user_adaptation in [False, "train", "test"]:
            raise Exception(
                "Possible values are:\n \
                - False (bool): No adaptation is performed. The train and test splits are completly disjoint. The adaptation split is empty.\n \
                - 'train' (str): A small percentage (defined in the config) of the annotations by test users is contained in the training split. The adaptation split is empty. This mirrors a situation in which one can obtain a minimal amount of annotationd *before* training the system.\n \
                - 'test' (str): A small percentage (defined in the config) of the annotations by the test user is in the adapatation split. This mirrors a situation in which one has a trained system (trained on the training users, with no annotations from the test users) and want to adapt the system *after* training it.\n"
                )
        
        self.user_adaptation = user_adaptation
        self.named = named
        self.extended = extended

        log.info("Generating. Named: %s, User adaptation: %s, Extended: %s" % (named, user_adaptation, extended))
        self.training_set = self.adaptation_set = self.test_set = None

        if not user_adaptation and not named:
            raise Exception("Invalid parameter configuration (user_adaptation=False, named=False). \
                            You need to at least know the explicit user traits for test users if no annotations are available")
        
        user_ids = set(list(self.dataset["rater_id"]))


        
        # Sample adapt+test users
        seed(config.seed)
        adaptation_test_user_ids = sample(sorted(user_ids), int(len(user_ids) * config.dataset_specific_splits[self.name]["user_based_split_percentage"]))
        train_user_ids = [u for u in user_ids if not u in adaptation_test_user_ids]
        adapt_test_text_id = [t_id for t_id, user in zip(self.dataset["text_id"], self.dataset["rater_id"]) if user in adaptation_test_user_ids]
        seed(config.seed)
        adaptation_text_ids = sample(sorted(adapt_test_text_id), int(len(adapt_test_text_id) * config.dataset_specific_splits[self.name]["text_based_split_percentage"]))
        test_text_ids = [t_id for t_id in adapt_test_text_id if t_id not in adaptation_text_ids]
        
        train_split , adaptation_split, test_split = PerspectivistSplit(type="train"), PerspectivistSplit(type="adaptation"), PerspectivistSplit(type="test")
        splits = [train_split, adaptation_split, test_split]
        for split in splits:
            for row in tqdm(self.dataset):
                # Read user
                if (row['rater_id'] in train_user_ids and split.type=="train") or \
                    (row['rater_id'] in adaptation_test_user_ids and split.type=="adaptation") or \
                      (row['rater_id'] in adaptation_test_user_ids and split.type=="test"):
                    if not row['rater_id'] in split.users:
                        split.users[row['rater_id']] = User(row['rater_id'])
                    
                    # Read traits only if named
                    if named:
                        split.users[row['rater_id']].traits["Gender"]=[row['rater_gender']]
                        if "Gender" in self.traits:
                            self.traits["Gender"].add(row["rater_gender"])
                        else:
                            self.traits["Gender"] = {(row["rater_gender"])}

                        split.users[row['rater_id']].traits["Generation"]=[row['rater_age']]
                        if "Generation" in self.traits:
                            self.traits["Generation"].add(row["rater_age"])
                        else:
                            self.traits["Generation"] = {(row["rater_age"])}
                
                # Read text
                if (row['rater_id'] in train_user_ids and split.type=="train") or \
                    (row['rater_id'] in adaptation_test_user_ids and row['text_id'] in adaptation_text_ids and split.type=="adaptation") or \
                        (row['rater_id'] in adaptation_test_user_ids and row['text_id'] in test_text_ids and split.type=="test"):
                    split.texts[row['text_id']] = {"context": row['context'], "reply": row['response']} 
                
                # Read annotation
                if (row['rater_id'] in train_user_ids and split.type=="train") or \
                    (row['rater_id'] in adaptation_test_user_ids and row['text_id'] in adaptation_text_ids and split.type=="adaptation") or \
                        (row['rater_id'] in adaptation_test_user_ids and row['text_id'] in test_text_ids and split.type=="test"):
                    split.annotation[(row['rater_id'], row['text_id'])] = {}
                    split.annotation[(row['rater_id'], row['text_id'])]["degree_of_harm"] = row['degree_of_harm']
                    self.labels["degree_of_harm"].add(row['degree_of_harm'])
                
                # Read labels by text
                if (row['rater_id'] in train_user_ids and split.type=="train") or \
                    (row['rater_id'] in adaptation_test_user_ids and row['text_id'] in adaptation_text_ids and split.type=="adaptation") or \
                        (row['rater_id'] in adaptation_test_user_ids and row['text_id'] in test_text_ids and split.type=="test"):
                    if not row['text_id'] in split.annotation_by_text:
                        split.annotation_by_text[row['text_id']] = []
                    split.annotation_by_text[row['text_id']].append(
                        {"user": split.users[row['rater_id']], "label": {"degree_of_harm": row['degree_of_harm']}})
                    self.labels["degree_of_harm"].add(row['degree_of_harm'])
        
        if user_adaptation == False:
            # You know nothing about the new test users except their explicit traits
            # You cannot use their adaptation annotations
            self.training_set = train_split
            self.adaptation_set = PerspectivistSplit(type="adaptation")
            self.test_set = test_split
                
        elif user_adaptation == "train":
            # You can use a few annotations by test users at training time
            # These annotations are directly included in the training split, 
            # the adaptation split is empty

            # Train + Adapt in the train set
            train_split.users = {**train_split.users, **adaptation_split.users}
            train_split.texts = {**train_split.texts, **adaptation_split.texts}
            train_split.annotation = {**train_split.annotation, **adaptation_split.annotation}

            for t_id in adaptation_split.annotation_by_text.keys():
                if t_id in train_split.annotation_by_text:
                    # add the annotatios
                    train_split.annotation_by_text[t_id] = train_split.annotation_by_text[t_id] + adaptation_split.annotation_by_text[t_id]
                else:
                    train_split.annotation_by_text[t_id] = adaptation_split.annotation_by_text[t_id]
            self.training_set = train_split
            self.adaptation_set = PerspectivistSplit(type="adaptation")
            self.test_set = test_split

                
        elif user_adaptation == "test":
            # You CANNOT use any test annotations at training time
            # However, you can use a few annotations to adapt your trained system to test users 
            # These adaptation annotations from test users are in the adaptation split, 
            self.training_set = train_split
            self.adaptation_set = adaptation_split
            self.test_set = test_split

        if not extended:
            strict_train_split = self.training_set
            strict_train_split.annotation_by_text = {t:self.training_set.annotation_by_text[t] for t in self.training_set.annotation_by_text if t not in self.test_set.annotation_by_text}
            # Filter annotations
            for u, t in copy.deepcopy(self.training_set.annotation):
                if t in self.test_set.annotation_by_text:
                    strict_train_split.annotation.pop((u, t))
    
            # Filter texts
            strict_train_split.texts = {k:self.training_set.texts[k] for k in self.training_set.texts if not k in self.test_set.texts}
            self.training_set = strict_train_split

        self.check_splits(user_adaptation, extended, named)
        self.describe_splits()
@dataclass
class MD(PerspectivistDataset):
    def __init__(self):
        super(MD, self).__init__()
        self.name = "MD"
        dataset = load_dataset("csv", data_files="data/MD-Agreement_dataset/MD_agreement.csv")
        self.dataset = dataset["train"]
        self.labels["offensiveness"] = set()
        self.label = config.dataset_label[self.name]

    def get_splits(self, extended, user_adaptation, named):
        if not user_adaptation in [False, "train", "test"]:
            raise Exception(
                "Possible values are:\n \
                - False (bool): No adaptation is performed. The train and test splits are completly disjoint. The adaptation split is empty.\n \
                - 'train' (str): A small percentage (defined in the config) of the annotations by test users is contained in the training split. The adaptation split is empty. This mirrors a situation in which one can obtain a minimal amount of annotationd *before* training the system.\n \
                - 'test' (str): A small percentage (defined in the config) of the annotations by the test user is in the adapatation split. This mirrors a situation in which one has a trained system (trained on the training users, with no annotations from the test users) and want to adapt the system *after* training it.\n"
                )

        log.info("Generating Named: %s, User adaptation: %s, Extended: %s" % (named, user_adaptation, extended))

        
        self.user_adaptation = user_adaptation
        self.named = named
        self.extended = extended

        self.training_set = self.adaptation_set = self.test_set = None

        if not user_adaptation and not named:
            raise Exception("Invalid parameter configuration (user_adaptation=False, named=False). \
                            You need to at least know the explicit user traits for test users if no annotations are available")
	
        
        user_ids = set(list(self.dataset['annotators']))

        # Sample adapt+test users
        seed(config.seed)
        adaptation_test_user_ids = sample(sorted(user_ids), int(len(user_ids) * config.dataset_specific_splits[self.name]["user_based_split_percentage"]))
        train_user_ids = [u for u in user_ids if not u in adaptation_test_user_ids]
        adapt_test_text_id = [t_id for t_id, user in zip(self.dataset["text_id"], self.dataset["annotators"]) if user in adaptation_test_user_ids]
        seed(config.seed)
        adaptation_text_ids = sample(sorted(adapt_test_text_id), int(len(adapt_test_text_id) * config.dataset_specific_splits[self.name]["text_based_split_percentage"]))
        test_text_ids = [t_id for t_id in adapt_test_text_id if t_id not in adaptation_text_ids]

        train_split , adaptation_split, test_split = PerspectivistSplit(type="train"), PerspectivistSplit(type="adaptation"), PerspectivistSplit(type="test")
        splits = [train_split, adaptation_split, test_split]
        for split in splits:
            for row in tqdm(self.dataset):
                # Read user
                if (row['annotators'] in train_user_ids and split.type=="train") or \
                    (row['annotators'] in adaptation_test_user_ids and split.type=="adaptation") or \
                      (row['annotators'] in adaptation_test_user_ids and split.type=="test"):
                    if not row['annotators'] in split.users:
                        split.users[row['annotators']] = User(row['annotators'])
                    
                    # Read traits only if named
                    if named:
                        raise Exception("Invalid parameter configuration. \
                            This dataset does not contain any information about the annotators.")
                            
                    
                # Read text
                if (row['annotators'] in train_user_ids and split.type=="train") or \
                    (row['annotators'] in adaptation_test_user_ids and row['text_id'] in adaptation_text_ids and split.type=="adaptation") or \
                        (row['annotators'] in adaptation_test_user_ids and row['text_id'] in test_text_ids and split.type=="test"):
                    split.texts[row['text_id']] = {"text": row['text']} 
                
                # Read annotation
                if (row['annotators'] in train_user_ids and split.type=="train") or \
                    (row['annotators'] in adaptation_test_user_ids and row['text_id'] in adaptation_text_ids and split.type=="adaptation") or \
                        (row['annotators'] in adaptation_test_user_ids and row['text_id'] in test_text_ids and split.type=="test"):
                    split.annotation[(row['annotators'], row['text_id'])] = {}
                    split.annotation[(row['annotators'], row['text_id'])]["offensiveness"] = row['annotations']
                    self.labels["offensiveness"].add(row['annotations'])

                # Read labels by text
                if (row['annotators'] in train_user_ids and split.type=="train") or \
                    (row['annotators'] in adaptation_test_user_ids and row['text_id'] in adaptation_text_ids and split.type=="adaptation") or \
                        (row['annotators'] in adaptation_test_user_ids and row['text_id'] in test_text_ids and split.type=="test"):
                    if not row['text_id'] in split.annotation_by_text:
                        split.annotation_by_text[row['text_id']] = []
                    split.annotation_by_text[row['text_id']].append(
                        {"user": split.users[row['annotators']], "label": {"offensiveness": row['annotations']}})
                    self.labels["offensiveness"].add(row['annotations'])
        
        if user_adaptation == False:
            # You know nothing about the new test users except their explicit traits
            # You cannot use their adaptation annotations
            self.training_set = train_split
            self.adaptation_set = PerspectivistSplit(type="adaptation")
            self.test_set = test_split
                
        elif user_adaptation == "train":
            # You can use a few annotations by test users at training time
            # These annotations are directly included in the training split, 
            # the adaptation split is empty

            # Train + Adapt in the train set
            train_split.users = {**train_split.users, **adaptation_split.users}
            train_split.texts = {**train_split.texts, **adaptation_split.texts}
            train_split.annotation = {**train_split.annotation, **adaptation_split.annotation}

            for t_id in adaptation_split.annotation_by_text.keys():
                if t_id in train_split.annotation_by_text:
                    # add the annotatios
                    train_split.annotation_by_text[t_id] = train_split.annotation_by_text[t_id] + adaptation_split.annotation_by_text[t_id]
                else:
                    train_split.annotation_by_text[t_id] = adaptation_split.annotation_by_text[t_id]
            self.training_set = train_split
            self.adaptation_set = PerspectivistSplit(type="adaptation")
            self.test_set = test_split

                
        elif user_adaptation == "test":
            # You CANNOT use any test annotations at training time
            # However, you can use a few annotations to adapt your trained system to test users 
            # These adaptation annotations from test users are in the adaptation split, 
            self.training_set = train_split
            self.adaptation_set = adaptation_split
            self.test_set = test_split

        if not extended:
            strict_train_split = self.training_set
            strict_train_split.annotation_by_text = {t:self.training_set.annotation_by_text[t] for t in self.training_set.annotation_by_text if t not in self.test_set.annotation_by_text}
            # Filter annotations
            for u, t in copy.deepcopy(self.training_set.annotation):
                if t in self.test_set.annotation_by_text:
                    strict_train_split.annotation.pop((u, t))
    
            # Filter texts
            strict_train_split.texts = {k:self.training_set.texts[k] for k in self.training_set.texts if not k in self.test_set.texts}
            self.training_set = strict_train_split

        self.check_splits(user_adaptation, extended, named)
        self.describe_splits()
