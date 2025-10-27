from perseval.config_marem import prompts
from perseval.check_marem import *
from perseval.data import *

dataset_name = "Epic"
label = "irony"

if dataset_name == "Epic":
    perspectivist_dataset = Epic(label)
elif dataset_name == "Brexit":
    perspectivist_dataset = Brexit()
elif dataset_name == "DICES":
    perspectivist_dataset = DICES(label)
perspectivist_dataset.get_splits(user_adaptation="train", extended=False, named=True)

PerspectivistLLM (perspectivist_dataset)