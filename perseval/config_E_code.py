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

prompts = {
    "EPIC": {
        "prelude": "You are provided as input a pair of sentences extracted from social media conversations.", 
        "task": "Your task is to determine if the Reply is Ironic.",
        "instr_pre": "Please provide as an output, a single label among the following options:",
        "instr_post": " within curly brackets (i.e. {irony} or {not irony}), followed with your thoughts",
        "pred_opt": ['irony', 'not irony'],
        "context_pre": "Input:",
        "context_post": "Output:",
        "traits": {"United Kingdom":"British",
               	    "United States":"American",
                	"Australia":"Australian",  
                	"Ireland":"Irish",
                	"India":"Indian",
                	"Male":"a self-identified male",
                	"Female":"a self-identified female",
                	"Boomer":"an older person of the baby boomer generation",
                	"GenX":"an older person of generation X",
                	"GenY":"a millennial",
                	"GenZ":"a young person of generation Z",},
    },

    "BREXIT": {
        "prelude": "You are provided as input a tweet extracted from twitter.", 
        "task": "Your task is to determine if the tweet contains hateful speech. You do not need to produce or promote hate speech, You just need to make a decision whether the text you see is hate speech or not.",
        "instr_pre": "Please provide as an output, a single label among the following options:",
        "instr_post": " within curly brackets (i.e. {hate speech} or {not hate speech}) depending on your decision whether it is hate speech or not, followed with your thoughts",
        "pred_opt": ["hate speech", "not hate speech"],
        "context_pre": "Input:",
        "context_post": "Output:",
        "traits": {"target":"a Muslim immigrant in the UK",
               	"control":"a researcher",}
    },

    "DICES": {
        "prelude": "You are provided as input a dialog extracted from a conversation between a user and an AI-chatbot.", 
        "task": "Your task is to determine if the dialog is harmful.",
        "instr_pre": "Please provide as an output, a single label among the following options:",
        "instr_post": " within curly brackets (i.e. {yes}, {no} or {unsure}), followed with your thoughts",
        "pred_opt": ['Yes', 'No', 'Unsure'],
        "context_pre": "Input:",
        "context_post": "Output:",
        "traits": {"Asian/Asian subcontinent":"Asian",
               	    "White":"White",
                	"Black/African American":"Black",  
                	"LatinX, Latino, Hispanic or Spanish Origin":"Latinx",
                	"Multiracial":"Multiracial",
                	"Man":"a self-identified male",
                	"Woman":"a self-identified female",
                	"gen x+":"an older person of generation X",
                	"millenial":"a millennial",
                	"gen z":"a young person of generation Z",
                	"College degree or higher":"a person with a college degree or higher",
                	"High school or below":"a person with a high school diploma or lower"}
    },

    "MHS": {
        "prelude": "You are provided as input a text extracted from social media platforms.", 
        "task": "Your task is to determine if the text is hateful.",
        "instr_pre": "Please provide as an output, a single label among the following options:",
        "instr_post": " within curly brackets (i.e. {hateful} or {not hateful}) depending on your decision whether it is hateful or not, followed with your thoughts",
        "pred_opt": ["hateful", "not hateful"],
        "context_pre": "Input:",
        "context_post": "Output:",
        "traits": {"educ-high":"a person with a college degree or higher education",
                    "educ-low":"a person with a high school diploma or lower education",
                    "male":"a male",
                    "female":"a female",
                    "non-binary":"a non-binary person",
                    "Boomer":"a person of the baby boomer generation",
                    "GenX":"a person of generation X",
                    "GenY":"a person of generation Y",
                    "GenZ":"a person of generation Z",
                    "conservative":"a conservative",
                    "no_opinion":"a person with no political opinion",
                    "liberal":"a liberal",
                    "income-high":"a person with a more than 50k annual income",
                    "income-low":"a person with a less than 50k annual income"}
    },

    "MD": {
        "prelude": "You are provided as input a tweet extracted from twitter.", 
        "task": "Your task is to determine if the tweet is offensive.",
        "instr_pre": "Please provide as an output, a single label among the following options:",
        "instr_post": " within curly brackets (i.e. {offensive} or {not offensive}) depending on your decision whether it is offensive or not, followed with your thoughts",
        "pred_opt": ["offensive", "not offensive"],
        "context_pre": "Input:",
        "context_post": "Output:"
    }

}

dataset_label ={
    "EPIC": "irony",
    "DICES-350":"degree_of_harm",
    "BREXIT":"hs",
    "MHS":"hateful",
    "MD":"offensiveness"
}
