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