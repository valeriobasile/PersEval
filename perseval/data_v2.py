
import pickle
import logging as log
from os.path import isfile
from random import seed, sample

from tqdm import tqdm
import numpy as np
from datasets import load_dataset, concatenate_datasets
from sklearn.model_selection import train_test_split
from . import config

log.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    encoding='utf-8', 
    level=log.INFO)

# changing the random seed will change how the datasets are split into training and test set
seed(config.seed)


class PerspectivistDataset:
    def __init__(self):
        self.name = None
        self.filename = None
        self.trait_set = set()
        self.labels = dict()
        self.training_set = None
        self.adaptation_set = None
        self.test_set = None

    def save(self):            
        log.info(f"Saving {self.name} to {self.filename}")
        with open(self.filename, "wb") as fo:
            pickle.dump((self.__dict__, self.training_set, self.test_set), fo)

    def load(self):            
        log.info(f"Loading {self.name} from file {self.filename}")
        with open(self.filename, "rb") as f:
            self.__dict__, self.training_set, self.test_set = pickle.load(f)

    def describe_splits(self):
        # TODO: check
        if not self.training_set:
            raise Exception("You need to first choose a task through "+self.name+".get_splits(task_name, named, strict)")
        
        print("--- Unique users ---")
        print("Train set: %d" % len(self.training_set.users))
        if len(self.adaptation_set.users):
            print("adaptation set: %d" % len(self.adaptation_set.users))
        print("Test set: %d" % len(self.test_set.users))
        print()
        print("--- Unique texts ---")
        print("Train set: %d" % len(self.training_set.annotation_by_text))
        if len(self.adaptation_set.annotation_by_text):
            print("adaptation set: %d" % len(self.adaptation_set.annotation_by_text))
        print("Test set: %d" % len(self.test_set.annotation_by_text))
        print()
        print("--- Instances (text + user) ---")
        print("Train set: %d" % len(self.training_set.annotation))
        if len(self.adaptation_set.annotation):
            print("adaptation set: %d" % len(self.adaptation_set.annotation))
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


    def check_splits(self, user_adaptation, strict, named):
        if user_adaptation == False or user_adaptation == "train":
            # The adaptation set is empty
            assert self.adaptation_set == PerspectivistSplit(type="adaptation")
        
        # Users
        if user_adaptation == False or user_adaptation == "test":
            # Train and adapt + test users have no overlap
            assert set(self.training_set.users).intersection(set(self.adaptation_set.users)) == set()
            assert set(self.training_set.users).intersection(set(self.test_set.users)) == set()
        if user_adaptation == "train" and not strict:
            # All test users are also in the training set
            assert set(self.training_set.users).union(set(self.test_set.users)) == set(self.training_set.users) 
        
        # Texts
        # adapt and test texts have no overlap
        assert set(self.adaptation_set.texts).intersection(set(self.test_set.texts)) == set()  
        
        if strict:
            # Strict train and test text have no overlap
            assert set(self.training_set.texts).intersection(set(self.test_set.texts)) == set()  
        
        log.info("All tests passed")


class Instance:
    def __init__(self, instance_id, instance_text, user, label):
        self.instance_id = instance_id
        self.instance_text = instance_text
        self.user = user
        self.label = label

    def __repr__(self):
        return f"{self.instance_id} {self.user} {self.label}"


class PerspectivistSplit:
    def __init__(self, type=None):
        self.type = type # Str, e.g., train, adaptation, test
        self.users = dict() 
        self.texts = dict()
        self.annotation = dict() #user, text, label (Instance + label)
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
    
    def __eq__(self, other):
        if self.type == other.type and \
            (self.users == other.users) and \
            (self.texts == other.texts) and \
            (self.annotation == other.annotation) and\
            (self.annotation_by_text == other.annotation_by_text):
            return True
        else:
            return False
        

class User:
    def __init__(self, user):
        self.id = user
        self.traits = set()

    def __repr__(self):
        return "User: " + str(self.id)
    
    def __lt__(self, other):
        return self.id < other.id
    
    def __eq__(self, other):
        if self.id == other.id and self.traits == other.traits:
            return True
        else:
            return False


class Epic(PerspectivistDataset):
    def __init__(self):
        super(Epic, self).__init__()
        self.name = "EPIC"
        self.filename = f"{self.name}{config.dataset_filename_suffix}"
        if isfile(self.filename):
            self.load()
            return
        dataset = load_dataset("Multilingual-Perspectivist-NLU/EPIC")
        self.dataset = dataset["train"]
        self.labels["irony"] = set()

    def get_splits(self, strict=True, user_adaptation=False, named=True):
        self.training_set = self.adaptation_set = self.test_set = None

        if not user_adaptation and not named:
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
        train_text_ids = [t_id for t_id in self.dataset["id_original"] if t_id not in adapt_test_text_id]

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
                        trait = f"Gender: {row['Sex']}"
                        split.users[row['user']].traits.add(trait)
                        self.trait_set.add(trait)

                        trait = f"Nationality: {row['Nationality']}"
                        split.users[row['user']].traits.add(trait)
                        self.trait_set.add(trait)

                        try:
                            trait = f"Age: {self.__convert_age(int(row['Age']))}"
                            split.users[row['user']].traits.add(trait)
                            self.trait_set.add(trait)
                        except ValueError as e:
                            pass
                    
                # Read text
                if (row['id_original'] in train_text_ids and split.type=="train") or \
                    (row['id_original'] in adaptation_text_ids and split.type=="adaptation") or \
                        (row['id_original'] in test_text_ids and split.type=="test"):
                    split.texts[row['id_original']] = {"post": row['parent_text'], "reply": row['text']} 
                
                # Read annotation
                if (row['user'] in train_user_ids and row['id_original'] in train_text_ids and split.type=="train") or \
                    (row['user'] in adaptation_test_user_ids and row['id_original'] in adaptation_text_ids and split.type=="adaptation") or \
                        (row['user'] in adaptation_test_user_ids and row['id_original'] in test_text_ids and split.type=="test"):                    
                    split.annotation[(row['user'], row['id_original'])] = {}
                    split.annotation[(row['user'], row['id_original'])]["irony"] = row['label']
                    self.labels["irony"].add(row['label'])

                # Read labels by text
                if (row['user'] in train_user_ids and row['id_original'] in train_text_ids and split.type=="train") or \
                    (row['user'] in adaptation_test_user_ids and row['id_original'] in adaptation_text_ids and split.type=="adaptation") or \
                        (row['user'] in adaptation_test_user_ids and row['id_original'] in test_text_ids and split.type=="test"):                    
                    if not row['id_original'] in split.annotation_by_text:
                        split.annotation_by_text[row['id_original']] = []
                    split.annotation_by_text[row['id_original']].append(
                        {"user": split.users[row['user']], "label": {"irony": row['label']}})
                    self.labels["irony"].add(row['label'])
                
        if strict:
            strict_train_split = train_split
            strict_train_split.annotation_by_text = {t:train_split.annotation_by_text[t] for t in train_split.annotation_by_text if t not in test_split.annotation_by_text}
            # Filter annotation and users
            for u, t in train_split.annotation:
                if t in test_split.annotation_by_text:
                    strict_train_split.annotation.update({(u, t): train_split.annotation[(u, t)]})
                    strict_train_split.users[u] = User(u)
            # Filter texts
            strict_train_split.texts = {k:train_split.texts[k] for k in train_split.texts if not k in test_split.texts}
            train_split = strict_train_split

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
                
        else:
            raise Exception("TODO: explain the possibilities")
                
        self.check_splits(user_adaptation, strict, named)
                    


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
