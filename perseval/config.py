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

label_map = {
    "irony": {"iro":1, "not":0},
    "irony_pred": {"irony":1, "not irony":0},
    "hs": {"hate speech":1, "not hate speech":0},
    "hs_pred": {"hate speech":1, "not hate speech":0},
    "Q2_harmful_content_overall": {"Yes":2, "Unsure":1, "No":0},
    "Q2_harmful_content_overall_pred": {"yes":2, "unsure":1, "no":0}
}

prompts = {
    "EPIC": {
        "prelude": "You are provided as input a pair of sentences extracted from social media conversations.", 
        "task": "Your task is to determine if the Reply is Ironic.",
        "instr_pre": "Please provide as an output, a single label among the following options:",
        "instr_post": ", followed with your thoughts",
        "pred_opt": ['irony', 'not irony'],
        "context_pre": "Input:",
        "context_post": "Output:",
        "traits": [
                "British",
               	"American",
                "Australian",  
                "Irish",
                "Indian",
                "a self-identified male",
                "a self-identified female",
                "a millennial",
                "a young person of generation Z",
                "an older person of the baby boomer generation",
                "an older person of generation X",
                "a young person",
                "an older person"
            ]
    },
    "BREXIT": {
        "prelude": "You are provided as input a tweet extracted from twitter.", 
        "task": "Your task is to determine if the tweet is hateful.",
        "instr_pre": "Please provide as an output, a single label among the following options:",
        "instr_post": ", followed with your thoughts",
        "pred_opt": ["hate speech", "not hate speech"],
        "context_pre": "Input:",
        "context_post": "Output:",
        "traits": [
                "a Muslim immigrant in the UK",
               	"a person"
            ]
    },
    "DICES": {
        "prelude": "You are provided as input a dialog extracted from a conversation between a user and an AI-chatbot.", 
        "task": "Your task is to determine if the dialog is harmful.",
        "instr_pre": "Please provide as an output, a single label among the following options:",
        "instr_post": ", followed with your thoughts",
        "pred_opt": ['Yes', 'No', 'Unsure'],
        "context_pre": "Input:",
        "context_post": "Output:",
        "traits": [
                "Asian",
               	"White",
                "Black",  
                "Latinx",
                "Multiracial"
                "a self-identified male",
                "a self-identified female",
                "a millennial",
                "a young person of generation Z",
                "an older person of generation X",
                "a person with a college degree or higher",
                "a person with a high school diploma or lower",
            ]
    }
}