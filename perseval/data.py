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
        

        self.adaptation_test_user_ids = None
        self.train_user_ids = None
        self.adaptation_text_ids = None
        self.test_text_ids  = None
        
    def available_labels(self):
        """Return the labels available for this dataset."""
        return list(self.labels.keys())

    def default_label(self):
        """Return the default label for this dataset."""
        return self.label

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
                if i['user']==u:
                    user_adapt_texts+=1
            for i in self.test_set.annotation:
                if i['user']==u:
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
                if i['user']==u:
                    user_train_texts+=1 
            for i in self.adaptation_set.annotation:
                if i['user']==u:
                    user_adapt_texts+=1
            for i in self.test_set.annotation:
                if i['user']==u:
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

    def belongs_split(self, row, split):
        return (row[self.key_user] in self.train_user_ids and split.type=="train") or \
            (row[self.key_user] in self.adaptation_test_user_ids and row[self.key_text] in self.adaptation_text_ids and split.type=="adaptation") or \
            (row[self.key_user] in self.adaptation_test_user_ids and row[self.key_text] in self.test_text_ids and split.type=="test")

    


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
        self.annotation = [] #user, text, label
        self.annotation_by_text = dict()
        self._seen = set()

    def __iter__(self):
        #for (user, instance_id), label in self.annotation.items()
        for row in self.annotation:
                yield Instance(
                    row['text'], 
                    self.texts[row['text']],
                    self.users[row['user']],
                    row['label'])

    def __len__(self):
        return len(self.annotation)

    def record_annotation(self, dataset, row, label):
        key = (row[dataset.key_user], row[dataset.key_text])
        if key in self._seen:
            return
        self._seen.add(key)
        self.annotation.append({'user': row[dataset.key_user], 'text': row[dataset.key_text], 'label' : label})

    def merge_splits(self,other_splits):
        assert not (self._seen & other_splits._seen)
        self.annotation = self.annotation + other_splits.annotation
        self._seen = self._seen | other_splits._seen

    def make_strict(self,dataset):
        test_text = dataset.test_set.annotation_by_text
        self.annotation = [
            a for a in dataset.training_set.annotation if a['text'] not in test_text
        ]
        self._seen = {(a['user'], a['text']) for a in self.annotation}

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
    def __init__(self, label):
        super(Epic, self).__init__()
        self.name = "EPIC"
        self.label = label
        dataset = load_dataset("Multilingual-Perspectivist-NLU/EPIC")
        self.dataset = dataset["train"]
        self.dataset = self.dataset.map(lambda x: {"label": config.label_map[label][x["label"]]})
        self.label = config.dataset_label[self.name]
        self.labels[label] = set()
        self.key_user = 'user'
        self.key_text = 'id_original'

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
        
        user_ids = set(list(self.dataset[self.key_user]))

        # Sample adapt+test users
        seed(config.seed)
        self.adaptation_test_user_ids = sample(sorted(user_ids), int(len(user_ids) * config.dataset_specific_splits[self.name]["user_based_split_percentage"]))
        self.train_user_ids = [u for u in user_ids if not u in self.adaptation_test_user_ids]
        adapt_test_text_id = [t_id for t_id, user in zip(self.dataset["id_original"], self.dataset["user"]) if user in self.adaptation_test_user_ids]
        seed(config.seed)
        self.adaptation_text_ids = sample(sorted(adapt_test_text_id), int(len(adapt_test_text_id) * config.dataset_specific_splits[self.name]["text_based_split_percentage"]))
        self.test_text_ids = [t_id for t_id in adapt_test_text_id if t_id not in self.adaptation_text_ids]

        train_split , adaptation_split, test_split = PerspectivistSplit(type="train"), PerspectivistSplit(type="adaptation"), PerspectivistSplit(type="test")
        splits = [train_split, adaptation_split, test_split]
        for split in splits:
            for row in tqdm(self.dataset):
                # Read user
                if (row[self.key_user] in self.train_user_ids and split.type=="train") or \
                    (row[self.key_user] in self.adaptation_test_user_ids and split.type=="adaptation") or \
                    (row[self.key_user] in self.adaptation_test_user_ids and split.type=="test"):
                    if not row[self.key_user] in split.users:
                        split.users[row[self.key_user]] = User(row[self.key_user])
                    
                    # Read traits only if named
                    if named:
                        split.users[row[self.key_user]].traits["Gender"]=[row['Sex']]
                        if "Gender" in self.traits:
                            self.traits["Gender"].add(row["Sex"])
                        else:
                            self.traits["Gender"] = {(row["Sex"])}

                        split.users[row[self.key_user]].traits["Nationality"]=[row['Nationality']]
                        if "Nationality" in self.traits:
                            self.traits["Nationality"].add(row["Nationality"])
                        else:
                            self.traits["Nationality"] = {(row["Nationality"])}
                        try:
                            generation = self.__convert_age(int(row['Age']))
                            split.users[row[self.key_user]].traits["Generation"]=[generation]
                            if "Generation" in self.traits:
                                self.traits["Generation"].add(generation)
                            else:
                                self.traits["Generation"] = {generation}
                        except ValueError as e:
                            split.users[row[self.key_user]].traits["Generation"]=["UNK"]
                    
                # Read text
                if (self.belongs_split(row,split)):
                    split.texts[row[self.key_text]] = {"post": row['parent_text'], "reply": row['text']} 
                
                # Read annotation
                if (self.belongs_split(row,split)):
                    split.record_annotation(self, row, {self.label: row['label']})
                    self.labels[self.label].add(row['label'])

                # Read labels by text
                if (self.belongs_split(row,split)):
                    if not row[self.key_text] in split.annotation_by_text:
                        split.annotation_by_text[row[self.key_text]] = []
                    split.annotation_by_text[row[self.key_text]].append(
                        {"user": split.users[row[self.key_user]], "label": {self.label: row['label']}})
                    self.labels[self.label].add(row['label'])
        
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

            train_split.merge_splits(adaptation_split)

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
            strict_train_split.make_strict(self)
            
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
        self.key_user = 'annotator_id'
        self.key_text = 'instance_id'



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
        self.train_user_ids, self.adaptation_test_user_ids = train_test_split(user_ids,
                                                        test_size=config.dataset_specific_splits[self.name]["user_based_split_percentage"],
                                                        random_state=config.seed,
                                                        shuffle=True, stratify=user_group)
        seed(config.seed)
        all_text_ids = list(set(self.dataset["instance_id"]))
        train_text_ids = sample(sorted(all_text_ids), int(len(all_text_ids)*config.dataset_specific_splits[self.name]["text_based_split_percentage_train"]))
        adaptation_test_text_ids = [t for t in all_text_ids if t not in train_text_ids]
        self.adaptation_text_ids = sample(sorted(adaptation_test_text_ids), int(len(adaptation_test_text_ids)*config.dataset_specific_splits[self.name]["text_based_split_percentage_dev"]))
        self.test_text_ids = [t for t in adaptation_test_text_ids if t not in self.adaptation_text_ids]

        train_split, adaptation_split, test_split = PerspectivistSplit(type="train"), PerspectivistSplit(type="adaptation"), PerspectivistSplit(type="test")
        splits = [train_split, adaptation_split, test_split]
        for split in splits:
            for row in tqdm(self.dataset):
                # Read user
                if (row[self.key_user] in self.train_user_ids and split.type=="train") or \
                    (row[self.key_user] in self.adaptation_test_user_ids and split.type=="adaptation") or \
                      (row[self.key_user] in self.adaptation_test_user_ids and split.type=="test"):
                    if not row[self.key_user] in split.users:
                        split.users[row[self.key_user]] = User(row[self.key_user])

                    # Read traits only if named
                    if named:
                        split.users[row[self.key_user]].traits["Group"]=[row['annotator_group']]
                        if "Group" in self.traits:
                            self.traits["Group"].add(row['annotator_group'])
                        else:
                            self.traits["Group"] = {row['annotator_group']}

                # Read text
                if self.belongs_split(row, split):
                    split.texts[row[self.key_text]] = {"tweet": row['tweet']}

                # Read annotation
                if self.belongs_split(row, split):
                    split.record_annotation(self, row, {label: row[label] for label in self.labels})
                    for label in self.labels:
                        self.labels[label].add(row[label])

                # Read labels by text
                if self.belongs_split(row, split):
                    if not row[self.key_text] in split.annotation_by_text:
                        split.annotation_by_text[row[self.key_text]] = []
                    labels_dict = {label: row[label] for label in self.labels}
                    split.annotation_by_text[row[self.key_text]].append(
                        {"user": split.users[row[self.key_user]], "label": labels_dict})
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
            train_split.merge_splits(adaptation_split)

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
            strict_train_split.make_strict(self)
    
            # Filter texts
            strict_train_split.texts = {k:self.training_set.texts[k] for k in self.training_set.texts if not k in self.test_set.texts}
            self.training_set = strict_train_split

        self.check_splits(user_adaptation, extended, named)
        self.describe_splits()
        
@dataclass
class DICES(PerspectivistDataset):
    def __init__(self, label):
        super(DICES, self).__init__()
        self.name = "DICES"
        self.label = label
        self.dataset = load_from_disk("data/diverse_safety_adversarial_dialog_350_enhanced")
        self.dataset = self.dataset.map(lambda x: {label: config.label_map[label][x[label]]})
        self.labels[label] = set()
        self.key_user = 'rater_id'
        self.key_text = 'text_id'


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
        
        user_ids = set(list(self.dataset["rater_id"]))


        # Sample adapt+test users
        seed(config.seed)
        self.adaptation_test_user_ids = sample(sorted(user_ids), int(len(user_ids) * config.dataset_specific_splits[self.name]["user_based_split_percentage"]))
        self.train_user_ids = [u for u in user_ids if not u in self.adaptation_test_user_ids]
        adapt_test_text_id = [t_id for t_id, user in zip(self.dataset["text_id"], self.dataset["rater_id"]) if user in self.adaptation_test_user_ids]
        seed(config.seed)
        self.adaptation_text_ids = sample(sorted(adapt_test_text_id), int(len(adapt_test_text_id) * config.dataset_specific_splits[self.name]["text_based_split_percentage"]))
        self.test_text_ids = [t_id for t_id in adapt_test_text_id if t_id not in self.adaptation_text_ids]

        train_split , adaptation_split, test_split = PerspectivistSplit(type="train"), PerspectivistSplit(type="adaptation"), PerspectivistSplit(type="test")
        splits = [train_split, adaptation_split, test_split]
        for split in splits:
            for row in tqdm(self.dataset):
                # Read user
                if (row[self.key_user] in self.train_user_ids and split.type=="train") or \
                    (row[self.key_user] in self.adaptation_test_user_ids and split.type=="adaptation") or \
                      (row[self.key_user] in self.adaptation_test_user_ids and split.type=="test"):
                    if not row[self.key_user] in split.users:
                        split.users[row[self.key_user]] = User(row[self.key_user])

                    # Read traits only if named
                    if named:
                        split.users[row[self.key_user]].traits["Gender"]=[row['rater_gender']]
                        if "Gender" in self.traits:
                            self.traits["Gender"].add(row["rater_gender"])
                        else:
                            self.traits["Gender"] = {(row["rater_gender"])}

                        split.users[row[self.key_user]].traits["Generation"]=[row['rater_age']]
                        if "Generation" in self.traits:
                            self.traits["Generation"].add(row["rater_age"])
                        else:
                            self.traits["Generation"] = {(row["rater_age"])}

                        split.users[row[self.key_user]].traits["Race"]=[row['rater_race']]
                        if "Race" in self.traits:
                            self.traits["Race"].add(row["rater_race"])
                        else:
                            self.traits["Race"] = {(row["rater_race"])}

                        split.users[row[self.key_user]].traits["Education"]=[row['rater_education']]
                        if "Education" in self.traits:
                            self.traits["Education"].add(row["rater_education"])
                        else:
                            self.traits["Education"] = {(row["rater_education"])}


                # Read text
                if self.belongs_split(row, split):
                    split.texts[row[self.key_text]] = {"context": row['context'], "reply": row['response']}

                # Read annotation
                if self.belongs_split(row, split):
                    split.record_annotation(self, row, {self.label: row[self.label]})
                    self.labels[self.label].add(row[self.label])

                # Read labels by text
                if self.belongs_split(row, split):
                    if not row[self.key_text] in split.annotation_by_text:
                        split.annotation_by_text[row[self.key_text]] = []
                    split.annotation_by_text[row[self.key_text]].append(
                        {"user": split.users[row[self.key_user]], "label": {self.label: row[self.label]}})
                    self.labels[self.label].add(row[self.label])
        
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
            train_split.merge_splits(adaptation_split)

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
            strict_train_split.make_strict(self)
    
            # Filter texts
            strict_train_split.texts = {k:self.training_set.texts[k] for k in self.training_set.texts if not k in self.test_set.texts}
            self.training_set = strict_train_split

        self.check_splits(user_adaptation, extended, named)
        self.describe_splits()


@dataclass
class MHS(PerspectivistDataset):
    def __init__(self, label):
        super(MHS, self).__init__()
        self.name = "MHS"
        self.label = label
        dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech")
        self.dataset = dataset["train"]
        self.dataset = self.dataset.map(lambda x: {"hateful": 1 if x["hatespeech"] > 0 else 0})
        self.labels[label] = set()
        self.key_user = 'annotator_id'
        self.key_text = 'comment_id'

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
        self.adaptation_test_user_ids = sample(sorted(user_ids), int(len(user_ids) * config.dataset_specific_splits[self.name]["user_based_split_percentage"]))
        self.train_user_ids = [u for u in user_ids if not u in self.adaptation_test_user_ids]
        adapt_test_text_id = [t_id for t_id, user in zip(self.dataset["comment_id"], self.dataset["annotator_id"]) if user in self.adaptation_test_user_ids]
        seed(config.seed)
        self.adaptation_text_ids = sample(sorted(adapt_test_text_id), int(len(adapt_test_text_id) * config.dataset_specific_splits[self.name]["text_based_split_percentage"]))
        self.test_text_ids = [t_id for t_id in adapt_test_text_id if t_id not in self.adaptation_text_ids]

        train_split , adaptation_split, test_split = PerspectivistSplit(type="train"), PerspectivistSplit(type="adaptation"), PerspectivistSplit(type="test")
        splits = [train_split, adaptation_split, test_split]
        education={"college_grad_aa":"educ-high","college_grad_ba":"educ-high","high_school_grad":"educ-low","masters":"educ-high","phd":"educ-high","professional_degree":"educ-low","some_college":"educ-low","some_high_school":"educ-low"}
        ideology={"conservative":"conservative","extremely_conservative":"conservative","extremely_liberal":"liberal","liberal":"liberal","neutral":"neutral","no_opinion":"neutral","slightly_conservative":"conservative","slightly_liberal":"liberal"}
        income={"100k-200k":"income-high","10k-50k":"income-low","<10k":"income-low",">200k":"income-high","50k-100k":"income-high"}
        for split in splits:
            for row in tqdm(self.dataset):
                # Read user
                if (row[self.key_user] in self.train_user_ids and split.type=="train") or \
                    (row[self.key_user] in self.adaptation_test_user_ids and split.type=="adaptation") or \
                    (row[self.key_user] in self.adaptation_test_user_ids and split.type=="test"):
                    if not row[self.key_user] in split.users:
                        split.users[row[self.key_user]] = User(row[self.key_user])

                    # Read traits only if named
                    if named:

                        # Education
                        if row['annotator_educ'] is not None:
                            split.users[row[self.key_user]].traits["Education"]=[education[row['annotator_educ']]]
                            if "Education" in self.traits:
                                self.traits["Education"].add(education[row["annotator_educ"]])
                            else:
                                self.traits["Education"] = {(education[row["annotator_educ"]])}

                        # Gender
                        split.users[row[self.key_user]].traits["Gender"]=[row['annotator_gender']]
                        if "Gender" in self.traits:
                            self.traits["Gender"].add(row["annotator_gender"])
                        else:
                            self.traits["Gender"] = {(row["annotator_gender"])}

                        # Ideology
                        if row['annotator_ideology'] is not None:
                            split.users[row[self.key_user]].traits["Ideology"]=[ideology[row['annotator_ideology']]]
                            if "Ideology" in self.traits:
                                self.traits["Ideology"].add(ideology[row["annotator_ideology"]])
                            else:
                                self.traits["Ideology"] = {(ideology[row["annotator_ideology"]])}

                        # Race
                        """
                        for race in ["asian","black","latinx","middle_eastern","native_american","pacific_islander","white","other"]:
                            split.users[row[self.key_user]].traits["Race-"+race.replace("_","-")]=[row['annotator_race_'+race]]
                            if "Race-"+race.replace("_","-") in self.traits:
                                self.traits["Race-"+race.replace("_","-")].add("yes" if row["annotator_race_"+race] == True else "no")
                            else:
                                self.traits["Race-"+race.replace("_","-")] = {("yes" if row["annotator_race_"+race] == True else "no")}
                        """

                        # Income
                        if row['annotator_income'] is not None:
                            split.users[row[self.key_user]].traits["Income"]=[income[row['annotator_income']]]
                            if "Income" in self.traits:
                                self.traits["Income"].add(income[row["annotator_income"]])
                            else:
                                self.traits["Income"] = {(income[row["annotator_income"]])}

                        # Age
                        if row['annotator_age'] is not None:
                            split.users[row[self.key_user]].traits["Age"]=[self.__convert_age(int(row['annotator_age']))]
                            if "Age" in self.traits:
                                self.traits["Age"].add(self.__convert_age(int(row['annotator_age'])))
                            else:
                                self.traits["Age"] = {(self.__convert_age(int(row['annotator_age'])))}

                # Read text
                if self.belongs_split(row, split):
                    split.texts[row[self.key_text]] = {"post": row['text']}

                # Read annotation
                if self.belongs_split(row, split):
                    hateful = 1 if row["hatespeech"] > 0 else 0
                    split.record_annotation(self, row, {"hateful": hateful})
                    self.labels["hateful"].add(hateful)

                # Read labels by text
                if self.belongs_split(row, split):
                    if not row[self.key_text] in split.annotation_by_text:
                        split.annotation_by_text[row[self.key_text]] = []
                    split.annotation_by_text[row[self.key_text]].append(
                        {"user": split.users[row[self.key_user]], "label": {"hateful":1 if row["hatespeech"] > 0 else 0}})
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
            train_split.merge_splits(adaptation_split)

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
            strict_train_split.make_strict(self)
    
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
class MD(PerspectivistDataset):
    def __init__(self, label):
        super(MD, self).__init__()
        self.name = "MD"
        self.label = label
        dataset = load_dataset("csv", data_files="data/MD-Agreement_dataset/MD_agreement.csv")
        self.dataset = dataset["train"]
        self.labels[label] = set()
        self.key_user = 'annotators'
        self.key_text = 'text_id'

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
	
        
        user_ids = set(list(self.dataset['annotators']))

        # Sample adapt+test users
        seed(config.seed)
        self.adaptation_test_user_ids = sample(sorted(user_ids), int(len(user_ids) * config.dataset_specific_splits[self.name]["user_based_split_percentage"]))
        self.train_user_ids = [u for u in user_ids if not u in self.adaptation_test_user_ids]
        adapt_test_text_id = [t_id for t_id, user in zip(self.dataset["text_id"], self.dataset["annotators"]) if user in self.adaptation_test_user_ids]
        seed(config.seed)
        self.adaptation_text_ids = sample(sorted(adapt_test_text_id), int(len(adapt_test_text_id) * config.dataset_specific_splits[self.name]["text_based_split_percentage"]))
        self.test_text_ids = [t_id for t_id in adapt_test_text_id if t_id not in self.adaptation_text_ids]

        train_split , adaptation_split, test_split = PerspectivistSplit(type="train"), PerspectivistSplit(type="adaptation"), PerspectivistSplit(type="test")
        splits = [train_split, adaptation_split, test_split]
        for split in splits:
            for row in tqdm(self.dataset):
                # Read user
                if (row[self.key_user] in self.train_user_ids and split.type=="train") or \
                    (row[self.key_user] in self.adaptation_test_user_ids and split.type=="adaptation") or \
                      (row[self.key_user] in self.adaptation_test_user_ids and split.type=="test"):
                    if not row[self.key_user] in split.users:
                        split.users[row[self.key_user]] = User(row[self.key_user])

                    # Read traits only if named
                    if named:
                        raise Exception("Invalid parameter configuration. \
                            This dataset does not contain any information about the annotators.")


                # Read text
                if self.belongs_split(row, split):
                    split.texts[row[self.key_text]] = {"text": row['text']}

                # Read annotation
                if self.belongs_split(row, split):
                    split.record_annotation(self, row, {"offensiveness": row['annotations']})
                    self.labels["offensiveness"].add(row['annotations'])

                # Read labels by text
                if self.belongs_split(row, split):
                    if not row[self.key_text] in split.annotation_by_text:
                        split.annotation_by_text[row[self.key_text]] = []
                    split.annotation_by_text[row[self.key_text]].append(
                        {"user": split.users[row[self.key_user]], "label": {"offensiveness": row['annotations']}})
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
            train_split.merge_splits(adaptation_split)

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
            strict_train_split.make_strict(self)
    
            # Filter texts
            strict_train_split.texts = {k:self.training_set.texts[k] for k in self.training_set.texts if not k in self.test_set.texts}
            self.training_set = strict_train_split

        self.check_splits(user_adaptation, extended, named)
        self.describe_splits()        
        
_DATASETS = {
    "epic": lambda: Epic(config.dataset_label["EPIC"]),
    "brexit": Brexit,
    "dices": lambda: DICES(config.dataset_label["DICES-350"]),
    "mhs": lambda: MHS(config.dataset_label["MHS"]),
    "md": lambda: MD(config.dataset_label["MD"]),
}

def available_datasets():
    """
    Return the names of all datasets available through the public API.
    """
    return list(_DATASETS.keys())

def download(dataset_name):
    """
    Download and initialize a dataset.

    Parameters
    ----------
    dataset_name : str
        Dataset name. Case-insensitive.

    Returns
    -------
    PerspectivistDataset
        Initialized dataset object.
    """
    if not isinstance(dataset_name, str):
        raise TypeError("dataset_name must be a string")

    dataset_name = dataset_name.lower()

    if dataset_name not in _DATASETS:
        available = ", ".join(available_datasets())
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available datasets: {available}"
        )

    return _DATASETS[dataset_name]()    

def download_and_split(dataset_name, user_adaptation=False, extended=False, named=False, baseline=False,):
    """
    Download a dataset and generate its requested task split.

    Parameters
    ----------
    dataset_name : str
        Dataset name. Case-insensitive.

    user_adaptation : bool or str
        False, "train", or "test".

    extended : bool
        Whether to allow training/test text overlap.

    named : bool
        Whether to use named user representations.

    baseline : bool
        Whether to allow the baseline configuration.

    Returns
    -------
    PerspectivistDataset
        Dataset with training, adaptation and test splits.
    """
    
    dataset = download(dataset_name)

    dataset.get_splits(
        extended=extended,
        user_adaptation=user_adaptation,
        named=named,
        baseline=baseline,
    )

    return dataset