
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

def get_dataset(dataset_name):
    if dataset_name == "EPIC":
        return Epic()
    else:
        log.error(f"Dataset {dataset_name} not found")
                

class PerspectivistDataset:
    def __init__(self):
        self.name = None
        self.filename = None
        self.trait_set = set()
        self.labels = dict()
        self.training_set = None
        self.development_set = None
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
        if len(self.development_set.users):
            print("Development set: %d" % len(self.development_set.users))
        print("Test set: %d" % len(self.test_set.users))
        print()
        print("--- Unique texts ---")
        print("Train set: %d" % len(self.training_set.annotation_by_text))
        if len(self.development_set.annotation_by_text):
            print("Development set: %d" % len(self.development_set.annotation_by_text))
        print("Test set: %d" % len(self.test_set.annotation_by_text))
        print()
        print("--- Instances (text + user) ---")
        print("Train set: %d" % len(self.training_set.annotation))
        if len(self.development_set.annotation):
            print("Development set: %d" % len(self.development_set.annotation))
        print("Test set: %d" % len(self.test_set.annotation))
        print()

        print("--- User-text train/development/test distribution ---")
        number_user_train_texts, number_user_dev_texts, number_user_test_texts = [], [], [] 
        
        for u in self.test_set.users:
            user_dev_texts, user_test_texts = 0, 0
            for i in self.development_set.annotation:
                if i[0]==u:
                    user_dev_texts+=1
            for i in self.test_set.annotation:
                if i[0]==u:
                    user_test_texts+=1
            number_user_dev_texts.append(user_dev_texts)
            number_user_test_texts.append(user_test_texts)
        print("The mean number of texts per users in the test set is %.3f" % np.mean(number_user_test_texts))
        if self.development_set != PerspectivistSplit(type=="development"):
            percentage_in_dev = [d/t for d, t in zip(number_user_dev_texts, number_user_test_texts)]
            print("The mean percentage of texts per users in the development set is %.3f" % np.mean(percentage_in_dev))
            print("The mean number of texts per users in the development set is %.3f" % np.mean(number_user_dev_texts))
            print("The min percentage of texts per users in the development set is %.3f (i.e. %.0f instances)" % (np.min(percentage_in_dev), np.min(number_user_dev_texts)))
            print("The max percentage of texts per users in the development set is %.3f (i.e. %.0f instances)" % (np.max(percentage_in_dev), np.max(number_user_dev_texts)))


    def check_splits(self, user_adaptation, strict, named):
        if user_adaptation == False or user_adaptation == "train":
            # The development set is empty
            assert self.development_set == PerspectivistSplit(type="development")
        
        # Users
        if user_adaptation == False or user_adaptation == "test":
            # Train and dev + test users have no overlap
            assert set(self.training_set.users).intersection(set(self.development_set.users)) == set()
            assert set(self.training_set.users).intersection(set(self.test_set.users)) == set()
        if user_adaptation == "train" and not strict:
            # All test users are also in the training set
            assert set(self.training_set.users).union(set(self.test_set.users)) == set(self.training_set.users) 
        
        # Texts
        # Dev and test texts have no overlap
        assert set(self.development_set.texts).intersection(set(self.test_set.texts)) == set()  
        
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
        self.type = type # Str, e.g., train, development, test
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

        if not user_adaptation and not named:
            raise Exception("Invalid parameter configuration (user_adaptation=False, named=False). \
                            You need to at least know the explicit user traits for test users if no annotations are available")
        
        user_ids = set(list(self.dataset['user']))

        # Sample developtment+test users
        seed(config.seed)
        development_test_user_ids = sample(sorted(user_ids), int(len(user_ids) * config.dataset_specific_splits[self.name]["user_based_split_percentage"]))
        train_user_ids = [u for u in user_ids if not u in development_test_user_ids]
        dev_test_text_id = [t_id for t_id, user in zip(self.dataset["id_original"], self.dataset["user"]) if user in development_test_user_ids]
        seed(config.seed)
        development_text_ids = sample(sorted(dev_test_text_id), int(len(dev_test_text_id) * config.dataset_specific_splits[self.name]["text_based_split_percentage"]))
        test_text_ids = [t_id for t_id in dev_test_text_id if t_id not in development_text_ids]
        train_text_ids = [t_id for t_id in self.dataset["id_original"] if t_id not in dev_test_text_id]

        train_split , development_split, test_split = PerspectivistSplit(type="train"), PerspectivistSplit(type="development"), PerspectivistSplit(type="test")
        splits = [train_split, development_split, test_split]
        for split in splits:
            for row in tqdm(self.dataset):
                # Read user
                if (row['user'] in train_user_ids and split.type=="train") or \
                    (row['user'] in development_test_user_ids and split.type=="development") or \
                      (row['user'] in development_test_user_ids and split.type=="test"):
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
                    (row['id_original'] in development_text_ids and split.type=="development") or \
                        (row['id_original'] in test_text_ids and split.type=="test"):
                    split.texts[row['id_original']] = {"post": row['parent_text'], "reply": row['text']} 
                
                # Read annotation
                if (row['user'] in train_user_ids and row['id_original'] in train_text_ids and split.type=="train") or \
                    (row['user'] in development_test_user_ids and row['id_original'] in development_text_ids and split.type=="development") or \
                        (row['user'] in development_test_user_ids and row['id_original'] in test_text_ids and split.type=="test"):                    
                    split.annotation[(row['user'], row['id_original'])] = {}
                    split.annotation[(row['user'], row['id_original'])]["irony"] = row['label']
                    self.labels["irony"].add(row['label'])

                # Read labels by text
                if (row['user'] in train_user_ids and row['id_original'] in train_text_ids and split.type=="train") or \
                    (row['user'] in development_test_user_ids and row['id_original'] in development_text_ids and split.type=="development") or \
                        (row['user'] in development_test_user_ids and row['id_original'] in test_text_ids and split.type=="test"):                    
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
            # You cannot use their development annotations
            self.training_set = train_split
            self.development_set = PerspectivistSplit(type="development")
            self.test_set = test_split
                
        elif user_adaptation == "train":
            # You can use a few annotations by test users at training time
            # These annotations are directly included in the training split, 
            # the development split is empty

            # Train + Dev in the train set
            train_split.users = {**train_split.users, **development_split.users}
            train_split.texts = {**train_split.texts, **development_split.texts}
            train_split.annotation = {**train_split.annotation, **development_split.annotation}

            for t_id in development_split.annotation_by_text.keys():
                if t_id in train_split.annotation_by_text:
                    # add the annotatios
                    train_split.annotation_by_text[t_id] = train_split.annotation_by_text[t_id] + development_split.annotation_by_text[t_id]
                else:
                    train_split.annotation_by_text[t_id] = development_split.annotation_by_text[t_id]
            self.training_set = train_split
            self.development_set = PerspectivistSplit(type="development")
            self.test_set = test_split

                
        elif user_adaptation == "test":
            # You CANNOT use any test annotations at training time
            # However, you can use a few annotations to adapt your trained system to test users 
            # These development annotations from test users are in the development split, 
            self.training_set = train_split
            self.development_set = development_split
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
