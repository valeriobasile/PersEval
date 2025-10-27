import sys
sys.path.append("..")
import os
import string 
import pandas as pd
import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, \
    Trainer, set_seed, TrainingArguments, pipeline
from datasets import Dataset
import torch
from sklearn.utils import compute_class_weight
import numpy as np
from tqdm import tqdm

from . import config_marem


class PerspectivistLLM():
    def __init__(self, persp_dataset):
        # self.model_id = model_identifier
        self.training_split = persp_dataset.training_set
        self.adaptation_split = persp_dataset.adaptation_set        
        self.test_split = persp_dataset.test_set
        # self.label = label
        self.traits = persp_dataset.traits
        self.named = persp_dataset.named
        self.user_adaptation = persp_dataset.user_adaptation
        self.extended = persp_dataset.extended
        self.dataset = persp_dataset.name 


        def create_prompt(text, profile):
            prompt_options = config_marem.prompts[self.dataset]
        
            #add perspective if it is set
            if self.named:
                prompt = f"You are {profile}.\n"
            else:
                prompt = ""

            #add intructions and explanations
            prompt = prompt + f"{prompt_options['prelude']} {prompt_options['task']} {prompt_options['instr_pre']}" 
            
            #dynamically add the options
            pred_option_count = len(prompt_options["pred_opt"])
            for i in range(pred_option_count-1): 
                prompt = prompt + f" '{prompt_options['pred_opt'][i]}'"
            prompt = prompt + f" or '{prompt_options['pred_opt'][-1]}' {prompt_options['instr_post']}.\n"

            #add the context part
            prompt = prompt + f"{prompt_options['context_pre']}\n"
            for key, value in text.items():
                prompt = prompt + f"{key}:\n {value}\n"
            prompt = prompt + f"{prompt_options['context_post']}"

            return prompt
        

        settings = config_marem.prompts[self.dataset]["traits"]

        all_user_traits = {}
        for user, user_class in self.test_split.users.items():
            traits = user_class.traits 
            store_traits = {}

            for trait,value in traits.items():
                trait_value = None

                for k,v in settings.items():
                    if value[0] == k:
                        trait_value = v
                        break
                if trait_value is None: 
                    trait_value = "UKN"
                
                store_traits[trait] = trait_value
            all_user_traits[user] = store_traits


        csv_files = {} 

        list_users = []
        list_txt_ids  = []
        list_traits = []
        try:
            for sample in tqdm(self.test_split, desc="Processing samples"):
                user_id = sample.user.id
                text_id = sample.instance_id
                text = sample.instance_text

                if user_id not in all_user_traits:
                    continue
                

                # Get the traits for the current user
                traits = all_user_traits[user_id]

                # Write data for each trait of the user
                for trait, profile in traits.items():
                    list_traits.append(trait)
                    # Open or retrieve the CSV file for this trait
                    if trait not in csv_files:
                        file = open(f"./new_csv_marem/{trait}_check_prompts.csv", "a")
                        writer = csv.DictWriter(file, fieldnames=["user_id", "text_id", "prompt"])
                        writer.writeheader()
                        csv_files[trait] = (file, writer)

                    file, writer = csv_files[trait]

                    prompt = create_prompt(text, profile)

                    writer.writerow({
                        "user_id": user_id,
                        "text_id": text_id,
                        "prompt": prompt
                    })

                    list_users.append(user_id)
                    list_txt_ids.append(text_id)

        finally:
            for file, _ in csv_files.values():
                file.close()
        print("number of test users: ",len(set(list_users)))
        print("number of test instances:  ",len(list_txt_ids)//len(set(list_traits)))
        print("numer of unique test texts: ",len(set(list_txt_ids)))