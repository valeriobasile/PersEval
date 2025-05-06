from collections import defaultdict
import csv
from glob import glob 
import random
import numpy as np
import os 
from itertools import combinations
import contextlib

from perseval.evaluation import *


seed = 42

def ensembled_ablation (folder_path, dataset, list_traits, lamp=False):
    random.seed = (seed)
    data = {}
    if not lamp:
        for trait in list_traits: 
            for prediction_file in glob(f"{folder_path}/predictions_{dataset}_True_train_False_{trait}.csv"):
                perspective = prediction_file.replace(".csv", "")
                print(perspective)
                with open(prediction_file, newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    data[perspective] = []  # Store multiple rows per perspective

                    for row in reader:
                        data[perspective].append({
                            "user_id": row["user_id"],
                            "text_id": row["text_id"],
                            "pred": row["label"]
                        })
    else: 
        for trait in list_traits:
            for prediction_file in glob(f"{folder_path}/edited_{dataset}_{trait}_True.csv"):
                perspective = prediction_file.replace(".csv", "")
                print(perspective)
                with open(prediction_file, newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    data[perspective] = []  # Store multiple rows per perspective

                    for row in reader:
                        data[perspective].append({
                            "user_id": row["user_id"],
                            "text_id": row["text_id"],
                            "pred": row["predictions"]
                        })


    user_text_labels = defaultdict(list)
    for perspective in data:
        for item in data[perspective]:
            user_text_labels[(item["user_id"], item["text_id"])].append(item["pred"])

    ensemble_dict = []
    for (user_id, text_id), labels in user_text_labels.items():
        label_counts = defaultdict(int) # Count the occurrences of each label
        for label in labels:
            label_counts[label] += 1
        
        majority_label = max(label_counts, key=label_counts.get)
        ensemble_dict.append({"user_id": user_id, "text_id": text_id, "pred": majority_label})

    suffix = "_".join(list_traits)
    dir_ablation = f"{folder_path}/ablation_files"
    if not os.path.exists(dir_ablation):
        os.mkdir(dir_ablation)
    file_path = f"{dir_ablation}/ensembled_predictions_{dataset}_{suffix}.csv"


    with open(file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["user_id", "text_id", "pred"])
        writer.writeheader()
        writer.writerows(ensemble_dict)

    return file_path



def dict_combinations (datasets):
    d_combinations = {}

    for dataset, trait in datasets.items():
        d_combinations[dataset] = {}  
        for r in range(1,len(trait)+1):
            key_name = f"{r}_traits"
            trait_combination = list(combinations(trait,r))
            d_combinations[dataset][key_name] = trait_combination

    return d_combinations



def results_ablation (dataset, models, d_combinations, test_set, label, lamp=False):
    for model in models:
        folder_path = f"./predictions_{model}"
        print("="*80)
        print (model)
        print("="*80)
        for k,v in d_combinations.items():
            if k == dataset:
                for r, combs in v.items():
                    for comb in combs:
                        list_traits = list(comb)
                        print(list_traits)
                        file_path = ensembled_ablation(folder_path, dataset, list_traits, lamp=lamp)
                        print("-"*20)
                        suffix = "_".join(list_traits)
                        output_file = f"./results_ablation/{model}/classification_report_{dataset}_{model}_{suffix}.txt"
                        evaluator = Evaluator(prediction_path=file_path,
                            test_set=test_set,
                            label=label)
                        with open(output_file, "w") as f:
                            with contextlib.redirect_stdout(f):
                                evaluator.global_metrics()
                                evaluator.annotator_level_metrics()
                                evaluator.text_level_metrics()
                                evaluator.trait_level_metrics()