from datasets import load_dataset
from random import seed, sample
from . import config
from tqdm import tqdm
import pickle
from os.path import isfile
import logging as log
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
        self.label_set = set()
        self.training_set = PerspectivistSplit()
        self.test_set = PerspectivistSplit()

    def save(self):            
        log.info(f"Saving {self.name} to {self.filename}")
        with open(self.filename, "wb") as fo:
            pickle.dump((self.__dict__, self.training_set, self.test_set), fo)

    def load(self):            
        log.info(f"Loading {self.name} from file {self.filename}")
        with open(self.filename, "rb") as f:
            self.__dict__, self.training_set, self.test_set = pickle.load(f)

class Instance:
    def __init__(self, instance_id, instance_text, user, label):
        self.instance_id = instance_id
        self.instance_text = instance_text
        self.user = user
        self.label = label

    def __repr__(self):
        return f"{self.instance_id} {self.user} {self.label}"

class PerspectivistSplit:
    def __init__(self):
        self.users = dict()
        self.instances = dict()
        self.annotation = dict()

    def __iter__(self):
        for (user, instance_id), label in self.annotation.items():
            yield Instance(
                instance_id, 
                self.instances[instance_id],
                self.users[user],
                label)

    def __len__(self):
        return len(self.annotation)

class User:
    def __init__(self, user):
        self.id = user
        self.traits = set()

    def __repr__(self):
        return self.id


class Epic(PerspectivistDataset):
    def __init__(self):
        super(Epic, self).__init__()
        self.name = "EPIC"
        self.filename = f"{self.name}{config.dataset_filename_suffix}"

        if isfile(self.filename):
            self.load()
            return

        dataset = load_dataset("Multilingual-Perspectivist-NLU/EPIC")

        log.info("Reading annotators")
        users = set()
        for row in tqdm(dataset['train']):
            users.add(row['user'])

        log.info("Splitting training and test set")
        test_users = sample(sorted(users), int(len(users) * config.train_test_split))
        
        for reading_test_set in [False, True]:
            if reading_test_set:
                split = self.test_set
            else:
                split = self.training_set

            log.info(f"Reading annotator traits (test set: {reading_test_set})")
            for row in tqdm(dataset['train']):
                if (not row['user'] in test_users and not reading_test_set) or (row['user'] in test_users and reading_test_set):
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
                        pass #print (e)

            log.info(f"Reading text instances (test set: {reading_test_set})")
            for row in tqdm(dataset['train']):
                if (not row['user'] in test_users and not reading_test_set) or (row['user'] in test_users and reading_test_set):
                    split.instances[row['id_original']] = row['text']

            log.info(f"Reading labels (test set: {reading_test_set})")
            for row in tqdm(dataset['train']):
                if (not row['user'] in test_users and not reading_test_set) or (row['user'] in test_users and reading_test_set):
                    split.annotation[(row['user'], row['id_original'])] = row['label']
                    self.label_set.add(row['label'])

        self.save()

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