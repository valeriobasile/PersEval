seed = 42

dataset_specific_splits = {
    "EPIC": {
        "user_based_split_percentage" : 0.2,
        "text_based_split_percentage" : 0.05,
    },
    "DICES": {
        "user_based_split_percentage" : 0.2,
        "text_based_split_percentage" : 0.05,
    },
    "BREXIT": {
        "user_based_split_percentage" : 0.2,
        "text_based_split_percentage_train" : 0.7,
        "text_based_split_percentage_dev" : 0.05,
    },
    "MHS": {
        "user_based_split_percentage" : 0.2,
        "text_based_split_percentage" : 0.05,
    },
    "MD":{
        "user_based_split_percentage" : 0.2,
        "text_based_split_percentage" : 0.05,  
    }
}

model_config = {
    "roberta-base": {
        "output_dir": "./results",
        "num_train_epochs": 5,
        "learning_rate": 5e-6,
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 32, 
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_strategy": "epoch",
        "overwrite_output_dir": True,
        "load_best_model_at_end": True,
        "report_to": None
    }
}

padding="max_length"
truncation=True
max_length=512

prediction_dir = "predictions"



dataset_label ={
    "EPIC": "irony",
    "DICES":"Q2_harmful_content_overall",
    "BREXIT":"hs",
    "MHS":"hateful",
    "MD":"offensiveness"
}

prediction_dir_llama = "predictions_llama"
prediction_dir_mixtral = "predictions_mixtral"

label_map = {
    "irony": {"iro":1, "not":0},
    "irony_pred": {"irony":1, "not irony":0},
    "hs": {"hate speech":1, "not hate speech":0},
    "hs_pred": {"hate speech":1, "not hate speech":0},
    "Q2_harmful_content_overall": {"Yes":2, "Unsure":1, "No":0},
    "Q2_harmful_content_overall_pred": {"yes":2, "unsure":1, "no":0},
    "hateful_pred": {"hateful":1, "not hateful":0},
    "offensiveness_pred": {"offensive":1, "not offensive":0}
}

# options for label:
# EPIC   -> ["irony"]
# BREXIT -> ["hs", "offensiveness", "aggressiveness", "stereotype"]
# DICES  -> ["degree_of_harm"]
# MHS    -> ["hateful"]
# MD     -> ["offensiveness"]

dataset_label ={
    "EPIC": "irony",
    "DICES-350":"degree_of_harm",
    "BREXIT":"hs",
    "MHS":"hateful",
    "MD":"offensiveness"
}

model_config = {
    "roberta-base": {
        "eval_strategy": "epoch",
        "greater_is_better":False,
        "learning_rate": 5e-6,
        "load_best_model_at_end": True,
        "logging_dir":"./logs",
        "logging_strategy": "epoch",
        "metric_for_best_model":"eval_loss",
        "num_train_epochs": 5,
        "output_dir": "./results",
        "overwrite_output_dir": True,
        "per_device_eval_batch_size": 32, 
        "per_device_train_batch_size": 16,
        "report_to": None,
        "save_strategy": "epoch",
        "save_total_limit": 1
    }
}

padding="max_length"
truncation=True
max_length=512

prediction_dir = "predictions"

eval_percentage = 0.2
data_lamp_dir="data_LaMP"