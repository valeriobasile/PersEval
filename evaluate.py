from perseval.evaluation import *
from perseval.data import *
import contextlib


# options for labels:
# EPIC   -> ["irony"]
# BREXIT -> ["hs", "offensiveness", "aggressiveness", "stereotype"]
# DICES  -> ["Q2_harmful_content_overall"]
# MHS    -> ["hateful"]
# MD     -> ["offensiveness"]

label = "offensiveness"
perspectivist_dataset = MD(label)
perspectivist_dataset.get_splits(user_adaptation="train", extended=False, named=False)


dataset = "MD"
# models = ["llama", "mixtral"]
models = ["lamp_mixtral"] #"lamp_llama"
for model in models:
    # file_path= f"./predictions_{model}/predictions_{dataset}_True_train_False_Group.csv"
    # file_path= f"./predictions_{model}/edited_{dataset}_Group_True.csv"

    # file_path= f"./predictions_{model}/predictions_{dataset}_False_train_False_zero.csv"
    file_path = f"./predictions_{model}/edited_{dataset}_False.csv"
    evaluator = Evaluator(prediction_path=file_path,
                        test_set=perspectivist_dataset.test_set,
                        label=perspectivist_dataset.label)


    output_file = f"./results/result_{model}_{dataset}_False.txt"
    with open(output_file, "w") as f:
        with contextlib.redirect_stdout(f):
            evaluator.global_metrics()
            evaluator.annotator_level_metrics()
            evaluator.text_level_metrics()
            # evaluator.trait_level_metrics()




