from perseval.data import *
from perseval.model import *
from perseval.evaluation import *
from transformers.utils import logging
import matplotlib.pyplot as plt
logging.set_verbosity_info() 
train=True
baseline=True

data = [Epic(),Brexit(),MHS(),DICES(),MD()]
if train:
    for dataset in data:
        print("Dataset",dataset.name)
        for named in [True, False]:
            for user_adaptation in [False, "train"]:
                for extended in [False, True]:
                    if user_adaptation == False and not named:
                        continue
                    print("-"*100)
                    print("Named" if named else "Unnamed","Adaptation:","None" if not user_adaptation else user_adaptation.capitalize(),"Extended:","No" if not extended else "Yes")
                    perspectivist_dataset = dataset
                    perspectivist_dataset.get_splits(user_adaptation=user_adaptation, extended=extended, named=named)
                    model = PerspectivistEncoder("roberta-base", 
                                                perspectivist_dataset, 
                                                label=perspectivist_dataset.label,
                                                baseline=False)
                    trainer = model.train()
                    logs = trainer.state.log_history
                    train_loss = [log['loss'] for log in logs if 'loss' in log]
                    eval_loss = [log['eval_loss'] for log in logs if 'eval_loss' in log]
                    print("Train Loss",train_loss)
                    print("Eval Loss",eval_loss)
                    plt.plot(train_loss, label="Training Loss")
                    plt.plot(eval_loss, label="Validation Loss")
                    plt.xlabel("Epochs")
                    plt.ylabel("Loss")
                    plt.legend()
                    plt.savefig("%s_%s_%s_%s.png" % (perspectivist_dataset.name, perspectivist_dataset.named, perspectivist_dataset.user_adaptation, perspectivist_dataset.extended),dpi=600)
                    plt.close()
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
if baseline:
    print("-"*100+"BASELINES"+"-"*100)
    for dataset in data:
        perspectivist_dataset = dataset
        perspectivist_dataset.get_splits(user_adaptation=False, extended=False, named=False,baseline=baseline)
        print("BASELINE for ",perspectivist_dataset.name)
        model = PerspectivistEncoder("roberta-base", 
                                    perspectivist_dataset, 
                                    label=perspectivist_dataset.label,
                                    baseline=baseline)
        trainer = model.train()
        model.predict(trainer) 
        evaluator = Evaluator(prediction_path="predictions/predictions_%s_baseline.csv" % (perspectivist_dataset.name),
                            test_set=perspectivist_dataset.test_set,
                            label=perspectivist_dataset.label)
        evaluator.global_metrics()
        evaluator.annotator_level_metrics()
        evaluator.text_level_metrics()
    
    

