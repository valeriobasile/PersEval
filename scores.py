from perseval.data import *
from perseval.model import *
from perseval.evaluation import *
from transformers.utils import logging
logging.set_verbosity_error() 

for named in [True, False]:
    for user_adaptation in [False, "train"]:
        for extended in [False, True]:
            if user_adaptation == False and not named:
                continue
            print("-"*100)
            print("Named" if named else "Unnamed","Adaptation:","None" if not user_adaptation else user_adaptation.capitalize(),"Extended:","No" if not extended else "Yes")
            perspectivist_dataset = MHS()
            perspectivist_dataset.get_splits(extended,user_adaptation,named)
            model = PerspectivistEncoder("roberta-base", 
                                        perspectivist_dataset, 
                                        label="hateful")
            trainer = model.train()
            model.predict(trainer) 
            evaluator = Evaluator(prediction_path="predictions/predictions_%s_%s_%s_%s.csv" % (perspectivist_dataset.name, perspectivist_dataset.named, perspectivist_dataset.user_adaptation, perspectivist_dataset.extended),
                                test_set=perspectivist_dataset.test_set,
                                label="hateful")
            evaluator.global_metrics()
            evaluator.annotator_level_metrics()
            evaluator.text_level_metrics()
            evaluator.trait_level_metrics()
