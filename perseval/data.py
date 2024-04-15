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
        self.label_set = set()
        self.training_set = PerspectivistSplit()
        self.development_set = PerspectivistSplit()
        self.test_set = PerspectivistSplit()

    def save(self):            
        log.info(f"Saving {self.name} to {self.filename}")
        with open(self.filename, "wb") as fo:
            pickle.dump((self.__dict__, self.training_set, self.test_set), fo)

    def load(self):            
        log.info(f"Loading {self.name} from file {self.filename}")
        with open(self.filename, "rb") as f:
            self.__dict__, self.training_set, self.test_set = pickle.load(f)

    
    def check_splits(self):
        # Users
        # Train and dev + test users have no overlap
        assert set(self.training_set.users).intersection(set(self.development_set.users)) == set()
        assert set(self.training_set.users).intersection(set(self.test_set.users)) == set()
        # Dev and test users have the same users
        # In theory, this might not be true in case of a very unfortunate split
        assert set(self.development_set.users).union(set(self.test_set.users)) == set(self.development_set.users) 

        # Texts
        # Dev and test texts have no overlap
        
        assert set(self.development_set.instances).intersection(set(self.test_set.instances)) == set()  
        
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
        self.users = dict() # why not a set?
        self.instances = dict()
        self.annotation = dict()
        self.annotation_by_instance = dict()

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
        return "User: " + self.id
    
    def __lt__(self, other):
        return self.id < other.id


class Epic(PerspectivistDataset):
    def __init__(self):
        super(Epic, self).__init__()
        self.name = "EPIC"
        self.filename = f"{self.name}{config.dataset_filename_suffix}"
        self.dummy_examples = [
            ("Hey there! Nice to see you Minnesota/ND Winter Weather", "yes"),
#            ("deputymartinski please do..i need the second hand embarrassment so desperatly on my phone", "yes"),
            ("@samcguigan544 You are not allowed to open that until Christmas day!", "no"),
#            ("That moment when you have so much stuff to do but you open @tumblr ... #productivity #tumblr", "yes"),
#            ("Over on CBS at noon is #BALvsMIA. FOX has #SEAvsPHI after the #Saints. #NFL", "no"),
#            ("@eXoAnnihilator @roIIerCosta @KSIOlajidebt tells someone to spell correctly while completely fucking up his tweet", "yes"),
#            ("My whole life is just \"oh ok\".", "no"),
#            ("Why am I wide awake right now..:face_without_mouth:", "no"),
#            ("My dads letting me drywall with him for Christmas. Just what I always wanted.", "yes"),
#            ("\"@NBCSportsRoc: Arizona Coyotes forge deal with BMW http://t.co/s7ZhDHk5li\" But, relocation!!!", "yes")
            ]
        
        if isfile(self.filename):
            self.load()
            return

        dataset = load_dataset("Multilingual-Perspectivist-NLU/EPIC")

        log.info("Reading annotators")
        user_ids = set()
        for row in tqdm(dataset['train']):
            user_ids.add(row['user'])

        log.info("Performing the user-based split")
        # Sample developtment+test users
        development_test_user_ids = sample(sorted(user_ids), int(len(user_ids) * config.dataset_specific_splits[self.name]["user_based_split_percentage"]))
        
        train_split , develpment_test_split = PerspectivistSplit(type="train"), PerspectivistSplit(type="development_test")
        user_based_splits = [train_split, develpment_test_split]
        for split in user_based_splits:
            log.info(f"Reading annotator traits (set: {split.type})")
            for row in tqdm(dataset['train']):
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
                        pass #print (e)

            log.info(f"Reading messages (set: {split.type})")
            for row in tqdm(dataset['train']):
                if (not row['user'] in development_test_user_ids and split.type=="train") or (row['user'] in development_test_user_ids and split.type!="train"):
                    split.instances[row['id_original']] = {"post": row['parent_text'], "reply": row['text']} 

            log.info(f"Reading individual labels (set: {split.type})")
            for row in tqdm(dataset['train']):
                if (not row['user'] in development_test_user_ids and split.type=="train") or (row['user'] in development_test_user_ids and split.type!="train"):
                    split.annotation[(row['user'], row['id_original'])] = row['label']
                    self.label_set.add(row['label'])
            
            log.info(f"Reading labels by instance (set: {split.type})")
            for row in tqdm(dataset['train']):
                if (not row['user'] in development_test_user_ids and split.type=="train") or (row['user'] in development_test_user_ids and split.type!="train"):
                    if not row['id_original'] in split.annotation_by_instance:
                        split.annotation_by_instance[row['id_original']] = []
                    split.annotation_by_instance[row['id_original']].append(
                        {"user": split.users[row['user']], "label": row['label']})
                    self.label_set.add(row['label'])
        self.training_set = train_split


        log.info("Performing the instance-based split")
        self.development_set, self.test_set = PerspectivistSplit(type="development"), PerspectivistSplit(type="test")

        # Sample which annotations will be in the dev and which in the test
        development_text_ids = sample(sorted(develpment_test_split.instances), int(len(develpment_test_split.instances) * config.dataset_specific_splits[self.name]["instance_based_split_percentage"]))
        self.development_set.instances = {k:develpment_test_split.instances[k] for k in development_text_ids}
        self.test_set.instances = {k:develpment_test_split.instances[k] for k in develpment_test_split.instances.keys() if k not in development_text_ids}
        
        # Annotations and users
        self.development_set.annotation, self.test_set.annotation = {}, {}
        self.development_set.users, self.test_set.users = {}, {}
        for u, t in develpment_test_split.annotation:
            if t in development_text_ids:
                self.development_set.annotation.update({(u, t): develpment_test_split.annotation[(u, t)]})
                self.development_set.users[u] = User(u)
            else:
                self.test_set.annotation.update({(u, t): develpment_test_split.annotation[(u, t)]})
                self.test_set.users[u] = User(u)
        
        # Annotation by instance
        self.development_set.annotation_by_instance, self.test_set.annotation_by_instance = {}, {}
        for t in develpment_test_split.annotation_by_instance:
            if t in development_text_ids:
                self.development_set.annotation_by_instance.update({t: develpment_test_split.annotation_by_instance[t]})
            else:
                self.test_set.annotation_by_instance.update({t: develpment_test_split.annotation_by_instance[t]})
        
        # A few checks
        self.check_splits()
        '''
        # We do not have a user -> instances dict, right? 
        for user in develpment_test_split.users:
            print(user)
            # Take all the ids of the texts of the annotations of the user 
            # (it would probably be better to have a set in the user's object)
            user_text_ids = [a for a in develpment_test_split.annotation if a[0]==user]
            # Sample some of them
            user_development_texts = sample(sorted(user_text_ids), int(len(user_text_ids) * config.dataset_specific_splits[self.name]["instance_based_split_percentage"]))
            # Take the corresponding texts for the dev
            user_development_texts = {a:develpment_test_split.annotation[a] for a in user_development_texts}
            # The rest of the user's annotations go to the test set
            user_test_instances = [a for a in develpment_test_split.annotation if a not in user_development_texts]
            user_test_annotations = {a:develpment_test_split.annotation[a] for a in user_test_instances}

            # Add instances to the splits
            user_development_instance_ids = [i[1] for i in user_develppment_annotations]
            develpment_split.instances.update({k:develpment_test_split.instances[k] for k in develpment_test_split.instances if k in user_development_instance_ids})
            user_test_instance_ids = [i[1] for i in user_test_annotations]
            test_split.instances.update({k:develpment_test_split.instances[k] for k in develpment_test_split.instances if k in user_test_instance_ids})

        print(len(develpment_split.instances))
        print(len(test_split.instances))
        
        
            #print(user_annotations)
        #print(develpment_test_split.users)
        #print(develpment_test_split.instances)
        #print(develpment_test_split.annotation)
        #print(develpment_test_split.annotation_by_instance)
        '''

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
