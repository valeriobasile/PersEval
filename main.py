from perseval.data import *
from perseval.model import *
from perseval.evaluation import *
from transformers.utils import logging
from perseval.prompts import EPIC, BREXIT, MHS_prompts, DICES_prompts, MD_Agreement
import argparse
import glob

logging.set_verbosity_error() 


# options for models
# mistralai/Mixtral-8x7B-Instruct-v0.1
# meta-llama/Meta-Llama-3.1-8B-Instruct

# options for labels:
# EPIC   -> ["irony"]
# BREXIT -> ["hs", "offensiveness", "aggressiveness", "stereotype"]
# DICES  -> ["Q2_harmful_content_overall"]
# MHS    -> ["hateful"]
# MD     -> ["offensiveness"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-name", 
        type=str, 
        required=False,
        default="Epic", 
        choices=['Epic', 'Brexit', 'DICES', 'MD', 'MHS'],
        help="Name of the dataset to use. Options: ['Epic', 'Brexit', 'DICES', 'MD', 'MHS']")
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
        "--type",
        type=str,
        required=True,
        choices=['encoder', 'llm', 'LaMP'],)
    parser.add_argument(
        "--named",
        action='store_true')
    parser.add_argument(
        "--extended",
        action='store_true')
    parser.add_argument(
        "--adaptation",
        type=str,
        required=False,
        default="train"
    )
    parser.add_argument(
        "--context",
        action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    # initialize the dataset
    if args.dataset_name == "Epic":
        perspectivist_dataset = Epic(args.label)
        prompts = EPIC
    elif args.dataset_name == "Brexit":
        perspectivist_dataset = Brexit()
        prompts = BREXIT
    elif args.dataset_name == "DICES":
        perspectivist_dataset = DICES(args.label)
        prompts = DICES_prompts
    elif args.dataset_name == "MHS":
        perspectivist_dataset = MHS(args.label)
        prompts = MHS_prompts
    elif args.dataset_name == "MD":
        perspectivist_dataset = MD(args.label)
        prompts = MD_Agreement
    perspectivist_dataset.get_splits(user_adaptation=args.adaptation, extended=args.extended, named=args.named)
    
    # create the model
    if args.type=='llm':
        model = PerspectivistLLM(args.model_name, 
                                perspectivist_dataset, 
                                label=args.label)
    elif args.type=='encoder':
        model = PerspectivistEncoder(args.model_name, 
                                perspectivist_dataset, 
                                label=args.label)
    elif args.type=='LaMP':
        model = PerspectivistLaMP(args.model_name, 
                                perspectivist_dataset, 
                                label=args.label,
                                context=args.context)
        model.rank_profile(prompts)
        model.merge_profile()
        model.evaluate_dataset()
        exit()
        
    #"""
    # train/test the model if it is not an LLM, test only if it is an LLM
    if args.type=='encoder':
        trainer = model.train()
        model.predict(trainer)  # <-- Predictions are saved in the "predictions" folder, 
    else:                       #     The file must contain three columns:
        model.predict()         #     "user_id", "text_id", "label"
    #"""

    #run the evaluations depending on the experiment type
    filenames = []
    if args.named:
        filename = f"predictions_{perspectivist_dataset.name}_{perspectivist_dataset.named}_"
        path = model.output_path + "/" + filename
        files = glob.glob(path + "**")
    else:
        files = []
        filename = "predictions_%s_%s_%s_%s_zero.csv" % (perspectivist_dataset.name, perspectivist_dataset.named, perspectivist_dataset.user_adaptation, perspectivist_dataset.extended)
        files.append(model.output_path + "/" + filename)

    if args.type=='llm':
        for filename in files:
            #filename = "predictions_%s_%s_%s_%s_%s.csv" % (perspectivist_dataset.name, perspectivist_dataset.named, perspectivist_dataset.user_adaptation, perspectivist_dataset.extended, trait)
            #filename = model.output_path + "/" + filename
            filenames.append(filename)
    else:
        filename = "predictions_%s_%s_%s_%s.csv" % (perspectivist_dataset.name, perspectivist_dataset.named, perspectivist_dataset.user_adaptation, perspectivist_dataset.extended)
        filename = model.output_path + "/" + filename
        filenames.append(filename)

    for pred_files in filenames:   
        print("--"*10)
        print(f"FOR THE FILE: {pred_files}")
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