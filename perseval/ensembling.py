from collections import defaultdict
import csv
from glob import glob 
import random
import numpy as np

seed = 42

def ensembled_predictions (folder_path, dataset, lamp=False):
    random.seed = (seed)
    data = {}
    if not lamp:
        for prediction_file in glob(f"{folder_path}/predictions_{dataset}_True_train_False_*.csv"):
            perspective = prediction_file.replace(f"{folder_path}/predictions_{dataset}_True_train_False_", "").replace(".csv", "")
            with open(prediction_file, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                data[perspective] = []  # Store multiple rows per perspective

                for row in reader:
                    data[perspective].append({
                        "user_id": row["user_id"],
                        "text_id": row["text_id"],
                        "label": row["label"]
                    })
    else: 
        for prediction_file in glob(f"{folder_path}/edited_{dataset}_*_True.csv"):
            perspective = prediction_file.replace(f"{folder_path}/{dataset}_*_True.csv", "").replace(".csv", "")
            print(perspective)
            with open(prediction_file, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                data[perspective] = []  # Store multiple rows per perspective

                for row in reader:
                    data[perspective].append({
                        "user_id": row["user_id"],
                        "text_id": row["text_id"],
                        "label": row["predictions"]
                    })


    user_text_labels = defaultdict(list)
    for perspective in data:
        for item in data[perspective]:
            user_text_labels[(item["user_id"], item["text_id"])].append(item["label"])

    ensemble_dict = []
    for (user_id, text_id), labels in user_text_labels.items():
        # Count the occurrences of each label
        label_counts = defaultdict(int)
        for label in labels:
            label_counts[label] += 1
        
        majority_label = max(label_counts, key=label_counts.get)
        ensemble_dict.append({"user_id": user_id, "text_id": text_id, "label": majority_label})

    file_path = f"{folder_path}/ensembled_predictions_{dataset}.csv"


    with open(file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["user_id", "text_id", "label"])
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