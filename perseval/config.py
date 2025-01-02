seed = 42

dataset_specific_splits = {
    "EPIC": {
        "user_based_split_percentage" : 0.2,
        "text_based_split_percentage" : 0.05,
    },
    "DICES-350": {
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