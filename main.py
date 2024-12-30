from perseval.data import *
from perseval.model import *
from perseval.evaluation import *
from transformers.utils import logging
logging.set_verbosity_info() 

perspectivist_dataset = MD()
perspectivist_dataset.get_splits(user_adaptation="train", extended=False, named=False)

# options for label:
# EPIC   -> ["irony"]
# BREXIT -> ["hs", "offensiveness", "aggressiveness", "stereotype"]
# DICES  -> ["degree_of_harm"]
# MHS    -> ["hateful"]
# MD     -> ["offensiveness"]

model = PerspectivistEncoder("roberta-base", 
                            perspectivist_dataset, 
                            label=perspectivist_dataset.label,
                            baseline=False)

trainer = model.train()
model.predict(trainer) # <-- Predictions are saved in the "predictions" folder, 
                       #     The file must contain three columns:
                       #     "user_id", "text_id", "label"


evaluator = Evaluator(prediction_path="predictions/predictions_%s_%s_%s_%s.csv" % (perspectivist_dataset.name, perspectivist_dataset.named, perspectivist_dataset.user_adaptation, perspectivist_dataset.extended),
                      test_set=perspectivist_dataset.test_set,
                      label=perspectivist_dataset.label)
evaluator.global_metrics()
evaluator.annotator_level_metrics()
evaluator.text_level_metrics()
evaluator.trait_level_metrics()

# You can also access the metrics from
#print(evaluator.global_metrics_dic)
#print(evaluator.annotator_level_metrics_dic)
#print(evaluator.annotator_based_macro_avg)
#print(evaluator.text_level_metrics_dic)
#print(evaluator.text_based_macro_avg)
#print(evaluator.trait_level_metrics_dic)
#print(evaluator.trait_based_macro_avg)
