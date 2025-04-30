text_context_template = {

"prompt_zero": "Instruction: You are provided in input (Input) a pair of sentences ({cntxt_name}, {txt_name}) extracted from {data_source}. \
Your task is to determine if the {txt_name} is {task_adj} in the context of the {cntxt_name}. Please provide in output (Output) a single label among the following array {labels}. \
Input: ",


"prompt_demographics": "You are {trait}. \
Instruction: You are provided in input (Input) a pair of sentences ({cntxt_name}, {txt_name}) extracted from {data_source}. Your task is to determine if the {txt_name} is {task_adj} in the context of the {cntxt_name}. Please provide in output (Output) a single label among the following array {labels}. \
Input: "
    }


EPIC = {"dataset_name": "Epic",
    	"txt_name": "reply",
    	"cntxt_name": "post",
    	"data_source": "social media conversations",
    	"task_adj": "ironic",
    	"labels": ["irony", "not irony"],
    	"traits":  {"United Kingdom":"British",
               	    "United States":"American",
                	"Australia":"Australian",  
                	"Ireland":"Irish",
                	"India":"Indian",
                	"Male":"a self-identified male",
                	"Female":"a self-identified female",
                	"Boomer":"an older person of the baby boomer generation",
                	"GenX":"an older person of generation X",
                	"GenY":"a millennial",
                	"GenZ":"a young person of generation Z",}
    	}
   	 
DICES_prompts = {"dataset_name": "Dices",
    	"txt_name": "reply",
    	"cntxt_name": "context",
    	"data_source": "conversations between a user and an AI-chatbot",
    	"task_adj": "harmful",
    	"labels": ['offensive', 'not offensive', 'unsure'],
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
                	"High school or below":"a person with a high school diploma or lower",
				}
        }


#################################################

text_only_template = {
"prompt_zero": "Instruction: You are provided in input (Input) a sentence ({txt_name}) extracted from {data_source}. Your task is to determine if the {txt_name} is {task_adj}. \
Please provide in output (Output) a single label among the following array {labels}. \
Input: ",


"prompt_demographics": "You are {trait}. \
Instruction: You are provided in input (Input) a sentence ({txt_name}) extracted from {data_source}. Your task is to determine if the {txt_name} is {task_adj}. \
Please provide in output (Output) a single label among the following array {labels}.\
Input: "
    }


BREXIT = {"dataset_name": "Brexit",
    	"txt_name": "tweet",
    	"data_source": "Twitter",
    	"task_adj": "hateful",
    	"labels": ["hate speech", "not hate speech"],
    	"traits": {"target":"a Muslim immigrant in the UK",
               	"control":"a researcher",
				}
    	}


MD_Agreement = {"dataset_name": "MD",
    	"txt_name": "text",
    	"data_source": "Twitter",
    	"task_adj": "offensive",
    	"labels": ["offensive", "not offensive"],
    	}
   	 

MHS_prompts = {"dataset_name": "MHS",
    	"txt_name": "post",
    	"data_source": "social media platforms",
    	"task_adj": "hateful",
    	"labels": ["hateful", "not hateful"],
    	"traits":{"educ-high":"a person with a college degree or higher education",
        	"educ-low":"a person with a high school diploma or lower education",
        	"male":"a self-identified male",
        	"female":"a self-identified female",
        	"non-binary":"a non-binary person",
            "Boomer":"an older person of the baby boomer generation",
            "GenX":"an older person of generation X",
            "GenY":"a millennial",
            "GenZ":"a young person of generation Z",
        	"conservative": "conservative",
        	"neutral":"a person with no political opinion",
        	"liberal":"liberal",
        	"asian":"Asian",
        	"black":"Black",
        	"latinx":"Latinx",
        	"middle_eastern":"Middle Eastern",
        	"native_american":"Native American",
        	"pacific_islander":"a Pacific islander",
        	"white":"White",
        	"income-low":"a person with a less than 50k annual income",
        	"income-high":"a person with a more than 50k annual income",
		}
    	}