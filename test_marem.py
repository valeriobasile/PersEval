from perseval.personalized_llms import PrepareData
from perseval.data import *
from perseval.prompts import EPIC, BREXIT, MHS, DICES, MD_Agreement

# options for label:
# EPIC   -> ["irony"]
# BREXIT -> ["hs", "offensiveness", "aggressiveness", "stereotype"]
# DICES  -> ["degree_of_harm"]
# MD -> ["offensiveness"]


perspectivist_dataset = Epic()
perspectivist_dataset.get_splits(user_adaptation="test", extended=False, named=True)

PrepareData(persp_dataset=perspectivist_dataset, dataset_config=EPIC, named=True, context=True)

print()
print("Json ready")