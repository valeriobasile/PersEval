from perseval.data import *
from perseval.model import *
from perseval.evaluation import *
from perseval.config import prompts
from transformers.utils import logging
import argparse

logging.set_verbosity_error() 


# options for models
# mistralai/Mixtral-8x7B-Instruct-v0.1
# meta-llama/Meta-Llama-3.1-8B-Instruct

# options for labels:
# EPIC   -> ["irony"]
# BREXIT -> ["hs", "offensiveness", "aggressiveness", "stereotype"]
# DICES  -> ["Q2_harmful_content_overall"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-name", 
        type=str, 
        required=False,
        default="Epic", 
        choices=['Epic', 'Brexit', 'DICES'],
        help="Name of the dataset to use. Options: ['Epic', 'Brexit', 'DICES']")
    parser.add_argument(
        "--model-name",
        type=str,
        required=False,
        default="roberta-base",
        help="Name of the transformer model to run inference on")
    parser.add_argument(
        "--label",
        type=str,
        required=False,
        default="irony")
    parser.add_argument(
        "--use-llm",
        action='store_true')
    parser.add_argument(
        "--named",
        action='store_true')
    parser.add_argument(
        "--extended",
        action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    # initialize the dataset
    if args.dataset_name == "Epic":
        perspectivist_dataset = Epic(args.label)
    elif args.dataset_name == "Brexit":
        perspectivist_dataset = Brexit()
    elif args.dataset_name == "DICES":
        perspectivist_dataset = DICES(args.label)
    perspectivist_dataset.get_splits(user_adaptation="train", extended=args.extended, named=args.named)

    # create the model
    if args.use_llm:
        model = PerspectivistLLM(args.model_name, 
                                perspectivist_dataset, 
                                label=args.label)
    else:
        model = PerspectivistEncoder(args.model_name, 
                                perspectivist_dataset, 
                                label=args.label)

    # train/test the model if it is not an LLM, test only if it is an LLM
    if not args.use_llm:
        trainer = model.train()
        model.predict(trainer)  # <-- Predictions are saved in the "predictions" folder, 
    else:                       #     The file must contain three columns:
        model.predict()         #     "user_id", "text_id", "label"


    #run the evaluations depending on the experiment type
    filnames = []
    if self.named:
        settings = config.prompts[perspectivist_dataset.name]["traits"]
    else:
        settings = ["zero"]
    if args.use_llm:
        for trait in setting:
            filename = prediction_path="predictions/predictions_%s_%s_%s_%s_%s.csv" % (perspectivist_dataset.name, perspectivist_dataset.named, perspectivist_dataset.user_adaptation, perspectivist_dataset.extended, trait)
            filenames.append(filename)
    else:
        filename = prediction_path="predictions/predictions_%s_%s_%s_%s.csv" % (perspectivist_dataset.name, perspectivist_dataset.named, perspectivist_dataset.user_adaptation, perspectivist_dataset.extended)
 

    for pred_files in filenames:   
        evaluator = Evaluator(pred_files,
                            test_set=perspectivist_dataset.test_set,
                            label=args.label)
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


if __name__ == "__main__":
    main()