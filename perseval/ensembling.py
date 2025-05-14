from collections import defaultdict, Counter
import csv
from glob import glob 
import random
import numpy as np
import os 
from itertools import combinations
import re 
from perseval.evaluation import * 


def get_majority_label(labels):
    counts = Counter(labels)
    max_count = max(counts.values())
    tied = [label for label, count in counts.items() if count == max_count]
    return random.choice(tied) if len(tied) > 1 else tied[0]

def ensembled_prediction (folder_path, dataset, list_traits, lamp=False, seed=42):
    random.seed(seed)
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
                        match = re.search(r'-?\d+',row["label"])
                        predclean = int(match.group())

                        data[perspective].append({
                            "user_id": row["user_id"],
                            "text_id": row["text_id"],
                            "pred": predclean
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
                        match = re.search(r'\d+',row["predictions"])
                        predclean = int(match.group())
                        
                        data[perspective].append({
                            "user_id": row["user_id"],
                            "text_id": row["text_id"],
                            "pred": predclean
                        })


    user_text_labels = defaultdict(list)
    for perspective in data:
        for item in data[perspective]:
            user_text_labels[(item["user_id"], item["text_id"])].append(item["pred"])

    ensemble_dict = []
    for (user_id, text_id), labels in user_text_labels.items():
        majority_label = get_majority_label(labels)
        ensemble_dict.append({"user_id": user_id, "text_id": text_id, "pred": int(majority_label)})

    suffix = "_".join(list_traits)
    file_path = f"{folder_path}/ensembled_predictions_{dataset}_{suffix}.csv"


    with open(file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["user_id", "text_id", "pred"])
        writer.writeheader()
        writer.writerows(ensemble_dict)

    return file_path


#check
# df_gender = pd.read_csv(f"{prediction_dir}/predictions_{dataset}_True_train_False_Gender.csv")
# df_generation = pd.read_csv(f"{prediction_dir}/predictions_{dataset}_True_train_False_Generation.csv")
# df_nationality = pd.read_csv(f"{prediction_dir}/predictions_{dataset}_True_train_False_Nationality.csv")
# df_gender = df_gender.astype(str)
# df_generation= df_generation.astype(str)
# df_nationality = df_nationality.astype(str)
# df = pd.DataFrame.from_dict(ensemble_dict)
# df_check = df.merge(df_gender[["user_id", "text_id", "label"]], on=["user_id", "text_id"], suffixes=("", "_gender"))
# df_check = df_check.merge(df_generation[["user_id", "text_id", "label"]], on=["user_id", "text_id"], suffixes=("","_generation"))
# df_check = df_check.merge(df_nationality[["user_id", "text_id", "label"]], on=["user_id", "text_id"], suffixes=("", "_nationality"))
# df_check.to_csv("check.csv")