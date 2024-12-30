from perseval.data import *
from perseval.model import *
from perseval.evaluation import *
from transformers.utils import logging
logging.set_verbosity_info() 
train=False
baseline=True
if train:
    for named in [True, False]:
        for user_adaptation in [False, "train"]:
            for extended in [False, True]:
                if user_adaptation == False and not named:
                    continue
                print("-"*100)
                print("Named" if named else "Unnamed","Adaptation:","None" if not user_adaptation else user_adaptation.capitalize(),"Extended:","No" if not extended else "Yes")
                perspectivist_dataset = MHS()
                perspectivist_dataset.get_splits(user_adaptation=user_adaptation, extended=extended, named=named)
                model_label="irony" if perspectivist_dataset.name == "EPIC" or perspectivist_dataset.name == "BREXIT" else "hateful"
                model = PerspectivistEncoder("roberta-base", 
                                            perspectivist_dataset, 
                                            label=model_label,
                                            baseline=False)
                trainer = model.train()
                model.predict(trainer) 
                evaluator = Evaluator(prediction_path="predictions/predictions_%s_%s_%s_%s.csv" % (perspectivist_dataset.name, perspectivist_dataset.named, perspectivist_dataset.user_adaptation, perspectivist_dataset.extended),
                                    test_set=perspectivist_dataset.test_set,
                                    label=model_label)
                evaluator.global_metrics()
                evaluator.annotator_level_metrics()
                evaluator.text_level_metrics()
                evaluator.trait_level_metrics()
if baseline:
    print("-"*100+"BASELINE"+"-"*100)
    data = [Epic(),Brexit(),MHS()]
    for dataset in data:
        perspectivist_dataset = dataset
        perspectivist_dataset.get_splits(user_adaptation=False, extended=False, named=False,baseline=baseline)
        model = PerspectivistEncoder("roberta-base", 
                                    perspectivist_dataset, 
                                    label=perspectivist_dataset.label,
                                    baseline=baseline)
        trainer = model.train()
        model.predict(trainer) 
        evaluator = Evaluator(prediction_path="predictions/predictions_%s_baseline.csv" % (perspectivist_dataset.name),
                            test_set=perspectivist_dataset.test_set,
                            label=perspectivist_dataset.label)
        print("BASELINE for ",perspectivist_dataset.name)
        evaluator.global_metrics()
        evaluator.annotator_level_metrics()
        evaluator.text_level_metrics()
    
    

