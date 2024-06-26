
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
        self.training_set = PerspectivistSplit(type="train")
        self.strict_training_set = PerspectivistSplit(type="train")
        self.development_set = PerspectivistSplit(type="validation")
        self.test_set = PerspectivistSplit(type="test")

    def save(self):            
        log.info(f"Saving {self.name} to {self.filename}")
        with open(self.filename, "wb") as fo:
            pickle.dump((self.__dict__, self.training_set, self.test_set), fo)

    def load(self):            
        log.info(f"Loading {self.name} from file {self.filename}")
        with open(self.filename, "rb") as f:
            self.__dict__, self.training_set, self.test_set = pickle.load(f)

    def describe_splits(self):
        print("--- Unique users ---")
        print("Train set: %d" % len(self.training_set.users))
        print("Strict train set: %d" % len(self.strict_training_set.users))
        print("Development set: %d" % len(self.development_set.users))
        print("Test set: %d" % len(self.test_set.users))
        print()
        print("--- Unique texts ---")
        print("Train set: %d" % len(self.training_set.annotation_by_text))
        print("Strict train set: %d (%d lost)" % (len(self.strict_training_set.annotation_by_text), len(self.training_set.annotation_by_text)-len(self.strict_training_set.annotation_by_text)))
        print("Development set: %d" % len(self.development_set.annotation_by_text))
        print("Test set: %d" % len(self.test_set.annotation_by_text))
        print()
        print("--- Instances (text + user) ---")
        print("Train set: %d" % len(self.training_set.annotation))
        print("Strict train set: %d (%d lost)" % (len(self.strict_training_set.annotation), len(self.training_set.annotation)-len(self.strict_training_set.annotation)))
        print("Development set: %d" % len(self.development_set.annotation))
        print("Test set: %d" % len(self.test_set.annotation))
        print()
        print("--- User-text validation/test distribution ---")
        number_user_dev_texts, number_user_test_texts = [], [] 
        for u in self.development_set.users:
            user_dev_texts, user_test_texts = 0, 0 
            for i in self.development_set.annotation:
                if i[0]==u:
                    user_dev_texts+=1
            for i in self.test_set.annotation:
                if i[0]==u:
                    user_test_texts+=1
            number_user_dev_texts.append(user_dev_texts)
            number_user_test_texts.append(user_test_texts)
        percentage_in_dev = [d/t for d, t in zip(number_user_dev_texts, number_user_test_texts)]
        print("The mean percentage of texts per users in the development set is %.3f" % np.mean(percentage_in_dev))
        print("The mean number of texts per users in the development set is %.3f" % np.mean(number_user_dev_texts))
        print("The mean number of texts per users in the test set is %.3f" % np.mean(number_user_test_texts))
        print("The min percentage of texts per users in the development set is %.3f (i.e. %.0f instances)" % (np.min(percentage_in_dev), np.min(number_user_dev_texts)))
        print("The max percentage of texts per users in the development set is %.3f (i.e. %.0f instances)" % (np.max(percentage_in_dev), np.max(number_user_dev_texts)))

    def check_splits(self):
        # Users
        # Train and dev + test users have no overlap
        assert set(self.training_set.users).intersection(set(self.development_set.users)) == set()
        assert set(self.training_set.users).intersection(set(self.test_set.users)) == set()
        
        # Dev and test users have the same users
        # In theory, this might not be true in case of a very unfortunate split for non-dense datasets
        # TODO: try stratification on the user
        #assert set(self.development_set.users).union(set(self.test_set.users)) == set(self.development_set.users) 

        # Texts
        # Dev and test texts have no overlap
        assert set(self.development_set.texts).intersection(set(self.test_set.texts)) == set()  
        # Strict train and test text have no overlap
        assert set(self.strict_training_set.texts).intersection(set(self.test_set.texts)) == set()  
        # All text in the strict training set are also in the training set
        assert set(self.strict_training_set.texts).intersection(set(self.training_set.texts)) == set(self.strict_training_set.texts)  
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
        dataset = dataset["train"]

        log.info("Reading annotators")
        user_ids = set(list(dataset['user']))

        log.info("Performing the user-based split")
        # Sample developtment+test users
        development_test_user_ids = sample(sorted(user_ids), int(len(user_ids) * config.dataset_specific_splits[self.name]["user_based_split_percentage"]))
        
        train_split , development_test_split = PerspectivistSplit(type="train"), PerspectivistSplit(type="development_test")
        user_based_splits = [train_split, development_test_split]
        
        for split in user_based_splits:
            log.info(f"Reading annotator traits (set: {split.type})")
            for row in tqdm(dataset):
                if (not row['user'] in development_test_user_ids and split.type=="train") or (row['user'] in development_test_user_ids and split.type!="train"):
                    if not row['user'] in split.users:
                        split.users[row['user']] = User(row['user'])

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

            log.info(f"Reading messages (set: {split.type})")
            for row in tqdm(dataset):
                if (not row['user'] in development_test_user_ids and split.type=="train") or (row['user'] in development_test_user_ids and split.type!="train"):
                    split.texts[row['id_original']] = {"post": row['parent_text'], "reply": row['text']} 

            log.info(f"Reading individual labels (set: {split.type})")
            self.labels["irony"] = set()
            for row in tqdm(dataset):
                if (not row['user'] in development_test_user_ids and split.type=="train") or (row['user'] in development_test_user_ids and split.type!="train"):
                    split.annotation[(row['user'], row['id_original'])] = {}
                    split.annotation[(row['user'], row['id_original'])]["irony"] = row['label']
                    self.labels["irony"].add(row['label'])
            
            log.info(f"Reading labels by text (set: {split.type})")
            for row in tqdm(dataset):
                if (not row['user'] in development_test_user_ids and split.type=="train") or (row['user'] in development_test_user_ids and split.type!="train"):
                    if not row['id_original'] in split.annotation_by_text:
                        split.annotation_by_text[row['id_original']] = []
                    split.annotation_by_text[row['id_original']].append(
                        {"user": split.users[row['user']], "label": {"irony": row['label']}})
                    self.labels["irony"].add(row['label'])
        self.training_set = train_split

        log.info("Performing the text-based split")
        self.development_set, self.test_set = PerspectivistSplit(type="development"), PerspectivistSplit(type="test")

        # Sample which annotations will be in the dev and which in the test
        development_text_ids = sample(sorted(development_test_split.texts), int(len(development_test_split.texts) * config.dataset_specific_splits[self.name]["text_based_split_percentage"]))
        self.development_set.texts = {k:development_test_split.texts[k] for k in development_text_ids}
        self.test_set.texts = {k:development_test_split.texts[k] for k in development_test_split.texts.keys() if k not in development_text_ids}
        
        # Annotations and users
        self.development_set.annotation, self.test_set.annotation = {}, {}
        self.development_set.users, self.test_set.users = {}, {}
        for u, t in tqdm(development_test_split.annotation):
            if t in development_text_ids:
                self.development_set.annotation.update({(u, t): development_test_split.annotation[(u, t)]})
                self.development_set.users[u] = User(u)
            else:
                self.test_set.annotation.update({(u, t): development_test_split.annotation[(u, t)]})
                self.test_set.users[u] = User(u)
        
        # Annotation by text
        self.development_set.annotation_by_text, self.test_set.annotation_by_text = {}, {}
        for t in tqdm(development_test_split.annotation_by_text):
            if t in development_text_ids:
                self.development_set.annotation_by_text.update({t: development_test_split.annotation_by_text[t]})
            else:
                self.test_set.annotation_by_text.update({t: development_test_split.annotation_by_text[t]})

    
        # Create strict training set (remove test texts only)
        log.info("Cleaning the training set from test texts")
        # Filter annotation_by_text
        self.strict_training_set.annotation_by_text = {t:self.training_set.annotation_by_text[t] for t in self.training_set.annotation_by_text if t not in self.test_set.annotation_by_text}
        # Filter annotation and users
        for u, t in self.training_set.annotation:
            if t in self.test_set.annotation_by_text:
                self.strict_training_set.annotation.update({(u, t): self.training_set.annotation[(u, t)]})
                self.strict_training_set.users[u] = User(u)
        # Filter texts
        self.strict_training_set.texts = {k:self.training_set.texts[k] for k in self.training_set.texts if not k in self.test_set.texts}
        
        # A few checks
        self.check_splits()
        # Print some stats
        self.describe_splits()
        #self.save()

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


class Brexit(PerspectivistDataset):
    def __init__(self):
        super(Brexit, self).__init__()
        self.name = "BREXIT"
        self.filename = f"{self.name}{config.dataset_filename_suffix}"
        if isfile(self.filename):
            self.load()
            return
        dataset = load_dataset("silvia-casola/BREXIT")
        dataset = concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])

        log.info("Reading annotators")
        users = set()
        for row in tqdm(dataset):
            users.add((row['annotator_id'], row['annotator_group']))
        user_ids = list([u[0] for u in users])
        user_group = list([u[1] for u in users]) 

        log.info("Performing the user-based split")
        # Sample developtment+test users
        _, development_test_user_ids = train_test_split(user_ids, 
                                                        test_size=config.dataset_specific_splits[self.name]["user_based_split_percentage"], 
                                                        random_state=config.seed, 
                                                        shuffle=True, stratify=user_group)
        
        # Sample texts
        all_text_ids = list(set(dataset["instance_id"]))
        train_text_ids = sample(sorted(all_text_ids), int(len(all_text_ids)*config.dataset_specific_splits[self.name]["text_based_split_percentage_train"]))
        development_test_text_ids = [t for t in all_text_ids if t not in train_text_ids]
        development_text_ids = sample(sorted(development_test_text_ids), int(len(development_test_text_ids)*config.dataset_specific_splits[self.name]["text_based_split_percentage_dev"]))
        test_text_ids = [t for t in development_test_text_ids if t not in development_text_ids]
        
        train_split, development_split, test_split = PerspectivistSplit(type="train"), PerspectivistSplit(type="development"), PerspectivistSplit(type="test")
        splits = [train_split, development_split, test_split]
        print(development_test_user_ids)
        for split in splits:
            log.info(f"Reading annotator traits (set: {split.type})")
            for row in tqdm(dataset):
                if (not row['annotator_id'] in development_test_user_ids and split.type=="train") or (row['annotator_id'] in development_test_user_ids and split.type!="train"):
                    if not row['annotator_id'] in split.users:
                        split.users[row['annotator_id']] = User(row['annotator_id'])

                    trait = f"Group: {row['annotator_group']}"
                    split.users[row['annotator_id']].traits.add(trait)
                    self.trait_set.add(trait)
        
            log.info(f"Reading messages (set: {split.type})")
            for row in tqdm(dataset):
                if (not row['annotator_id'] in development_test_user_ids and split.type=="train" and (row["instance_id"] in train_text_ids or row["instance_id"] in test_text_ids)) or \
                    (row['annotator_id'] in development_test_user_ids and split.type=="development" and row["instance_id"] in development_text_ids) or \
                        (row['annotator_id'] in development_test_user_ids and split.type=="test" and row["instance_id"] in test_text_ids):
                        split.texts[row['instance_id']] = row['tweet']

            log.info(f"Reading individual labels (set: {split.type})")
            labels = ["hs", "offensiveness", "aggressiveness", "stereotype"]
            for label in labels:
                self.labels[label] = set()
            for row in tqdm(dataset):
                if (not row['annotator_id'] in development_test_user_ids and split.type=="train" and (row["instance_id"] in train_text_ids or row["instance_id"] in test_text_ids)) or \
                    (row['annotator_id'] in development_test_user_ids and split.type=="development" and row["instance_id"] in development_text_ids) or \
                        (row['annotator_id'] in development_test_user_ids and split.type=="test" and row["instance_id"] in test_text_ids):
                    split.annotation[(row['annotator_id'], row['instance_id'])] = {}
                    for label in labels:
                        split.annotation[(row['annotator_id'], row['instance_id'])][label] = row[label]
                        self.labels[label].add(row[label])
            
            log.info(f"Reading labels by text (set: {split.type})")
            for row in tqdm(dataset):
                if (not row['annotator_id'] in development_test_user_ids and split.type=="train" and (row["instance_id"] in train_text_ids or row["instance_id"] in test_text_ids)) or \
                    (row['annotator_id'] in development_test_user_ids and split.type=="development" and row["instance_id"] in development_text_ids) or \
                        (row['annotator_id'] in development_test_user_ids and split.type=="test" and row["instance_id"] in test_text_ids):
                    if not row['instance_id'] in split.annotation_by_text:
                        split.annotation_by_text[row['instance_id']] = []
                    labels_dict = {label: row[label] for label in labels}    
                    split.annotation_by_text[row['instance_id']].append(
                        {"user": split.users[row['annotator_id']], "label": labels_dict})
                    for label in labels:
                        self.labels[label].add(row[label])
        self.training_set = train_split
        #self.strict_training_set = train_split
        self.development_set = development_split
        self.test_set = test_split

        # Create strict training set (remove test texts only)
        log.info("Cleaning the training set from test texts")

        # Filter annotation_by_text
        self.strict_training_set.annotation_by_text = {t:self.training_set.annotation_by_text[t] for t in self.training_set.annotation_by_text if t not in self.test_set.annotation_by_text}

        # Filter annotation and users
        for u, t in self.training_set.annotation:
            if t in self.test_set.annotation_by_text:
                self.strict_training_set.annotation.update({(u, t): self.training_set.annotation[(u, t)]})
                self.strict_training_set.users[u] = User(u)

        # Filter texts
        self.strict_training_set.texts = {k:self.training_set.texts[k] for k in self.training_set.texts if not k in self.test_set.texts}

        # A few checks
        self.check_splits()

        # Print some stats
        self.describe_splits()

        #self.save()


class MHS(PerspectivistDataset):
    def __init__(self):
        super(MHS, self).__init__()
        self.name = "MHS"
        self.filename = f"{self.name}{config.dataset_filename_suffix}"
        if isfile(self.filename):
            self.load()
            return
        dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech")
        dataset = dataset["train"]

        log.info("Reading annotators")
        user_ids = set(list(dataset['annotator_id']))

        log.info("Performing the user-based split")
        # Sample developtment+test users
        development_test_user_ids = sample(sorted(user_ids), int(len(user_ids) * config.dataset_specific_splits[self.name]["user_based_split_percentage"]))
        
        train_split, development_test_split = PerspectivistSplit(type="train"), PerspectivistSplit(type="development_test")
        user_based_splits = [train_split, development_test_split]
        
        for split in user_based_splits:
            log.info(f"Reading annotator traits (set: {split.type})")
            for row in tqdm(dataset):
                if (not row['annotator_id'] in development_test_user_ids and split.type=="train") or (row['annotator_id'] in development_test_user_ids and split.type!="train"):
                    if not row['annotator_id'] in split.users:
                        split.users[row['annotator_id']] = User(row['annotator_id'])
                    
                    trait = f"Gender: {row['annotator_gender']}"
                    split.users[row['annotator_id']].traits.add(trait)
                    self.trait_set.add(trait)
                    
                    trait = f"Transgender: {row['annotator_trans']}"
                    split.users[row['annotator_id']].traits.add(trait)
                    self.trait_set.add(trait)
                    
                    trait = f"Education: {row['annotator_educ']}"
                    split.users[row['annotator_id']].traits.add(trait)
                    self.trait_set.add(trait)
                    
                    trait = f"Income: {row['annotator_income']}"
                    split.users[row['annotator_id']].traits.add(trait)
                    self.trait_set.add(trait)
                    
                    trait = f"Ideology: {row['annotator_ideology']}"
                    split.users[row['annotator_id']].traits.add(trait)
                    self.trait_set.add(trait)
                    
                    for r in ["annotator_race_asian", "annotator_race_black", "annotator_race_latinx", "annotator_race_middle_eastern", "annotator_race_native_american", "annotator_race_pacific_islander", "annotator_race_white", "annotator_race_other"]:
                        if row[r]==True:
                            race = r.split("_")[-1]        
                            trait = f"Race: {race}"
                            split.users[row['annotator_id']].traits.add(trait)
                            self.trait_set.add(trait)
                    
                    trait = f"Age: {row['annotator_age']}"
                    split.users[row['annotator_id']].traits.add(trait)
                    self.trait_set.add(trait)
                    
                    for r in ["annotator_religion_atheist", "annotator_religion_buddhist", "annotator_religion_christian", "annotator_religion_hindu", "annotator_religion_jewish", "annotator_religion_mormon", "annotator_religion_muslim", "annotator_religion_nothing", "annotator_religion_other"]:
                        if row[r]==True:
                            religion = r.split("_")[-1]
                            trait = f"Religion: {religion}"
                            split.users[row['annotator_id']].traits.add(trait)
                            self.trait_set.add(trait)
                    
                    for r in ["annotator_sexuality_bisexual", "annotator_sexuality_gay", "annotator_sexuality_straight", "annotator_sexuality_other"]:
                        if row[r]==True:
                            sexuality = r.split("_")[-1]
                            trait = f"Sexuality: {sexuality}"
                            split.users[row['annotator_id']].traits.add(trait)
                            self.trait_set.add(trait)

            log.info(f"Reading messages (set: {split.type})")
            for row in tqdm(dataset):
                if (not row['annotator_id'] in development_test_user_ids and split.type=="train") or (row['annotator_id'] in development_test_user_ids and split.type!="train"):
                    split.texts[row['comment_id']] = {"post": row["text"]} 

            log.info(f"Reading individual labels (set: {split.type})")
            labels = ["hatespeech", "hate_speech_score", "sentiment", "respect", "insult", "humiliate", "status", "dehumanize", "violence", "genocide", "attack_defend"]
            for label in labels:
                self.labels[label] = set()
            for row in tqdm(dataset):
                if (not row['annotator_id'] in development_test_user_ids and split.type=="train") or (row['annotator_id'] in development_test_user_ids and split.type!="train"):
                    for label in labels:
                        split.annotation[(row['annotator_id'], row['comment_id'])] = {}
                        split.annotation[(row['annotator_id'], row['comment_id'])][label] = row[label]
                        self.labels[label].add(row[label])
      
            log.info(f"Reading labels by text (set: {split.type})")
            for row in tqdm(dataset):
                if (not row['annotator_id'] in development_test_user_ids and split.type=="train") or (row['annotator_id'] in development_test_user_ids and split.type!="train"):
                    if not row['comment_id'] in split.annotation_by_text:
                        split.annotation_by_text[row['comment_id']] = []
                    labels_dict = {label: row[label] for label in labels}
                    split.annotation_by_text[row['comment_id']].append(
                        {"annotator_id": split.users[row['annotator_id']], "label": labels_dict})
                    for label in labels:
                        self.labels[label].add(row[label])
        self.training_set = train_split

        log.info("Performing the text-based split")
        self.development_set, self.test_set = PerspectivistSplit(type="development"), PerspectivistSplit(type="test")

        # Sample which annotations will be in the dev and which in the test
        development_text_ids = sample(sorted(development_test_split.texts), int(len(development_test_split.texts) * config.dataset_specific_splits[self.name]["text_based_split_percentage"]))
        self.development_set.texts = {k:development_test_split.texts[k] for k in development_text_ids}
        self.test_set.texts = {k:development_test_split.texts[k] for k in development_test_split.texts.keys() if k not in development_text_ids}
        
        # Annotations and users
        self.development_set.annotation, self.test_set.annotation = {}, {}
        self.development_set.users, self.test_set.users = {}, {}
        for u, t in tqdm(development_test_split.annotation):
            if t in development_text_ids:
                self.development_set.annotation.update({(u, t): development_test_split.annotation[(u, t)]})
                self.development_set.users[u] = User(u)
            else:
                self.test_set.annotation.update({(u, t): development_test_split.annotation[(u, t)]})
                self.test_set.users[u] = User(u)
        
        # Annotation by text
        self.development_set.annotation_by_text, self.test_set.annotation_by_text = {}, {}
        for t in tqdm(development_test_split.annotation_by_text):
            if t in development_text_ids:
                self.development_set.annotation_by_text.update({t: development_test_split.annotation_by_text[t]})
            else:
                self.test_set.annotation_by_text.update({t: development_test_split.annotation_by_text[t]})
    
        # Create strict training set (remove test texts only)
        log.info("Cleaning the training set from test texts")
        # Filter annotation_by_text
        self.strict_training_set.annotation_by_text = {t:self.training_set.annotation_by_text[t] for t in self.training_set.annotation_by_text if t not in self.test_set.annotation_by_text}
        # Filter annotation and users
        for u, t in self.training_set.annotation:
            if t in self.test_set.annotation_by_text:
                self.strict_training_set.annotation.update({(u, t): self.training_set.annotation[(u, t)]})
                self.strict_training_set.users[u] = User(u)
        # Filter texts
        self.strict_training_set.texts = {k:self.training_set.texts[k] for k in self.training_set.texts if not k in self.test_set.texts}
        
        # A few checks
        self.check_splits()
        # Print some stats
        self.describe_splits()
        #self.save()