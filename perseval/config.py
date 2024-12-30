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
        "output_dir": "./results",
        "num_train_epochs": 5,
        "learning_rate": 5e-6,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 32, 
        "evaluation_strategy": "epoch",
        "save_strategy": "no",
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

eval_percentage = 0.2