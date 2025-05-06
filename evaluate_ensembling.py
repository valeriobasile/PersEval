from perseval.evaluation import *
from perseval.data import *
from perseval.ensembling import ensembled_predictions

# options for labels:
# EPIC   -> ["irony"]
# BREXIT -> ["hs", "offensiveness", "aggressiveness", "stereotype"]
# DICES  -> ["Q2_harmful_content_overall"]
# MHS    -> ["hateful"]
# MD     -> ["offensiveness"]

label = "Q2_harmful_content_overall"
perspectivist_dataset = DICES(label)
perspectivist_dataset.get_splits(user_adaptation="train", extended=False, named=False)


folder_path = "/home/marem/VScProjects/PersEval/predictions_lamp_llama"
dataset = "Dices"



file_path = ensembled_predictions(folder_path, dataset, lamp=True)
# file_path= f"{folder_path}/predictions_{dataset}_False_train_False_zero.csv"
# file_path = f"{folder_path}/{dataset}_False.csv"
evaluator = Evaluator(prediction_path=file_path,
                      test_set=perspectivist_dataset.test_set,
                      label=perspectivist_dataset.label,
                      ensembled=True)



evaluator.global_metrics()
evaluator.annotator_level_metrics()
evaluator.text_level_metrics()
evaluator.trait_level_metrics()





