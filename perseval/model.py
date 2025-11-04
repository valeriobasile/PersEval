import sys
sys.path.append("..")
import os
import string 

import pandas as pd
import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, \
    Trainer, set_seed, TrainingArguments, pipeline, AutoModel
from datasets import Dataset
import torch
from sklearn.utils import compute_class_weight
import numpy as np
from tqdm import tqdm
from perseval.personalized_llms import PrepareData
import json
import datasets

from . import config

class PerspectivistEncoder():
    def __init__(self, model_identifier, persp_dataset, label):
        self.model_id = model_identifier
        self.training_split = persp_dataset.training_set
        self.adaptation_split = persp_dataset.adaptation_set        
        self.test_split = persp_dataset.test_set
        self.label = label
        self.traits = persp_dataset.traits
        self.named = persp_dataset.named
        self.user_adaptation = persp_dataset.user_adaptation
        self.extended = persp_dataset.extended
        self.dataset = persp_dataset.name 
        self.output_path = config.prediction_dir
    
        self.tokenizer = AutoTokenizer.from_pretrained(model_identifier)
        if self.dataset == "DICES-350":
            self.model = AutoModelForSequenceClassification.from_pretrained(model_identifier, num_labels=4)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_identifier, num_labels=2)


    def train(self):
        self.__add_special_tokens_to_tokenizer()
        set_seed(config.seed)
        data = {"train" : self.__generate_data(self.training_split)[0]}   
        # computer class weight (in case labels are unbalanced)        
        try:
            class_weights = compute_class_weight(
                "balanced",
                classes=np.unique(data["train"]["labels"].float().numpy()),
                y=data["train"]["labels"].float().numpy()).astype("float32")
        except Exception as e:
            print("Unable to balance classes")
            class_weights = np.array([1, 1]).astype("float32")


        print('We will use the device:', torch.cuda.get_device_name(0))
        training_args = TrainingArguments(
            seed=config.seed,
            output_dir=config.model_config[self.model_id]["output_dir"],
            num_train_epochs=config.model_config[self.model_id]["num_train_epochs"],
            learning_rate=config.model_config[self.model_id]["learning_rate"],
            per_device_train_batch_size=config.model_config[self.model_id]["per_device_train_batch_size"],
            save_strategy=config.model_config[self.model_id]["save_strategy"],
            logging_strategy=config.model_config[self.model_id]["logging_strategy"],
            overwrite_output_dir=config.model_config[self.model_id]["overwrite_output_dir"],
            report_to=config.model_config[self.model_id]["report_to"]
        )


        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=data["train"],
        )
        trainer.set_class_weights(class_weights)
        trainer.train()
        return trainer
    

    def predict(self, trainer):
        test_data, ids = self.__generate_data(self.test_split)
        predictions = trainer.predict(test_data)
        if not os.path.exists(config.prediction_dir): 
            os.makedirs(config.prediction_dir)  
        with open(config.prediction_dir+"/predictions_%s_%s_%s_%s.csv" % (self.dataset, self.named, self.user_adaptation, self.extended), "w") as fo:
            writer = csv.DictWriter(
                fo,
                fieldnames=[
                    "user_id",
                    "text_id",
                    "label"])
            writer.writeheader()
            for i, id in zip(enumerate(test_data), ids):
                pred = np.argmax(predictions.predictions[i[0]])
                writer.writerow({
                    "user_id": id[0],
                    "text_id": id[1],
                    "label": pred
                })


    def __generate_data(self, split):
        ids, texts, labels = [], [], []
        for ann in split.annotation:
            ids.append(ann)
            texts.append(self.__add_special_tokens_to_text(split.users[ann[0]], split.texts[ann[1]]))
            labels.append(split.annotation[ann][self.label])
        df = pd.DataFrame({"text":texts, "labels":labels})
        dt = Dataset.from_pandas(df)
        tokenized_dataset = dt.map(lambda x: self.__tokenize(x, self.tokenizer), remove_columns=['text'])
        tokenized_dataset = tokenized_dataset.with_format("torch")
        return tokenized_dataset, ids


    def __tokenize(self, x, tokenizer):
        return tokenizer(
            x["text"], 
            padding=config.padding,
            truncation=config.truncation,
            max_length=config.max_length,
        )


    def __add_special_tokens_to_text(self, user, text):
        traits = sorted(list(self.traits.keys()))
        enriched_text = ""
        enriched_text +='<{}>'.format(user.id) + " "
        for trait in traits:
            if trait in user.traits:
                enriched_text +='<{}:{}>'.format(trait, user.traits[trait][0]) + " "
        for k in text:
            enriched_text += k + ": " + text[k] + " "
        return enriched_text
    
    
    def __add_special_tokens_to_tokenizer(self):
        special_tokens_dict = {"additional_special_tokens": []}
        new_tokens = set()
        for split in [self.training_split, self.adaptation_split, self.test_split]:
            for user in split.users:
                new_tokens.add('<{}>'.format(user))
                if self.named:
                    for dim in self.traits:
                        for trait in list(self.traits[dim]):
                            new_tokens.add('<{}:{}>'.format(dim, trait))
        special_tokens_dict['additional_special_tokens'] = list(new_tokens)        
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer)) 


class CustomTrainer(Trainer):
    """ Custom Trainer class to implement a custom loss function
    """

    def set_class_weights(self, class_weights):
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Focal Loss: https://arxiv.org/abs/1708.02002
        """
        gamma = 5.0
        alpha = .2
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.class_weights)).to("cuda")
        BCEloss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        # Focal Loss
        pt = torch.exp(-BCEloss)  # prevents nans when probability
        loss = alpha * (1 - pt) ** gamma * BCEloss
        return (loss, outputs) if return_outputs else loss


class PerspectivistLLM():
    def __init__(self, model_identifier, persp_dataset, label):
        self.model_id = model_identifier
        self.training_split = persp_dataset.training_set
        self.adaptation_split = persp_dataset.adaptation_set        
        self.test_split = persp_dataset.test_set
        self.label = label
        self.traits = persp_dataset.traits
        self.named = persp_dataset.named
        self.user_adaptation = persp_dataset.user_adaptation
        self.extended = persp_dataset.extended
        self.dataset = persp_dataset.name 
    
        if self.model_id == "mistralai/Mixtral-8x7B-Instruct-v0.1":
            self.mixtral = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.float16, device_map="auto")
            self.mixtral_tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.output_path = config.prediction_dir_mixtral
        elif self.model_id == "meta-llama/Meta-Llama-3.1-8B-Instruct":
            self.output_path = config.prediction_dir_llama
            self.llama = pipeline(
                "text-generation",
                model=self.model_id,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device="cuda")


    def predict(self):
        #prep output path
        if not os.path.exists(self.output_path): 
            os.makedirs(self.output_path)
        
        #prep experiment
        all_user_traits = {}
        if self.named:
            settings = config.prompts[self.dataset]["traits"]
            all_user_traits = self.get_user_traits(settings)
            
        #inference for each perspective
        csv_files = {}
        try:
            for sample in tqdm(self.test_split, desc="Processing samples"):
                text_id = sample.instance_id
                user_id = sample.user.id
                text = sample.instance_text

                if self.named:
                    if user_id not in all_user_traits:
                        continue
                    # Get the traits for the current user
                    traits = all_user_traits[user_id]
                else:
                    traits = {"zero": "zero"}

                #PATCH: TO BE DELETED LATER
                '''
                if user_id == 9226:
                    traits = {"Ideology": "from an unknown Ideology", "Education": "from an unknown Education", "Age": "from an unknown Age", "Income": "from an unknown Income"}
                elif user_id == 3797 or user_id == 6749:
                    traits = {"Income": "from an unknown Income"}
                else:
                    continue
                '''
                #PATCH ENDS HERE

                # Write data for each trait of the user
                for trait, profile in traits.items():
                    filename = "/predictions_%s_%s_%s_%s_%s.csv" % (self.dataset, self.named, self.user_adaptation, self.extended, trait)

                    # Open or retrieve the CSV file for this trait
                    if trait not in csv_files:
                        fo = open(self.output_path+filename, "a")
                        writer = csv.DictWriter(fo, fieldnames=["user_id", "text_id", "label", "thoughts", "prompt"])
                        writer.writeheader()
                        csv_files[trait] = (fo, writer)

                    fo, writer = csv_files[trait]

                    prompt = self.create_prompt(text, profile)
                    messages = [{"role": "user", "content": prompt}]


                    if self.model_id == "mistralai/Mixtral-8x7B-Instruct-v0.1":
                        #prepare
                        model_inputs = self.mixtral_tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
                        #infer
                        generated_ids = self.mixtral.generate(model_inputs, max_new_tokens=100, do_sample=False)
                        llm_response = self.mixtral_tokenizer.batch_decode(generated_ids)[0]
                    elif self.model_id == "meta-llama/Meta-Llama-3.1-8B-Instruct":
                        #infer
                        outputs = self.llama(messages, max_new_tokens=100, do_sample=False)
                        llm_response = outputs[0]["generated_text"][-1]["content"]

                    prediction = self.parse_output(llm_response)
                    
                    writer.writerow({
                        "user_id": user_id,
                        "text_id": text_id,
                        "label": prediction,
                        "thoughts": llm_response,
                        "prompt": prompt
                    })
        finally:
            for file, _ in csv_files.values():
                file.close()



    def create_prompt(self, text, trait):
        prompt_options = config.prompts[self.dataset]
      
        #add perspective if it is set
        if self.named:
            prompt = f"You are {trait}.\n"
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



    def parse_output(self, llm_response):
        #prep label strings
        pred_labels = config.label_map[self.label+"_pred"]

        #parse llm predictions
        if self.model_id == "mistralai/Mixtral-8x7B-Instruct-v0.1":
            llm_response = llm_response.split("[/INST]")[-1]
        pred_str = llm_response.split("{")[-1].split("}")[0].strip().lower()

        #filtering
        pred_str = pred_str.strip("\'")

        #map the predictions
        if pred_str in pred_labels:
            pred_int = pred_labels[pred_str]
        else:
            pred_int = -1

        return pred_int



    def get_user_traits(self, settings):
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
                    trait_value = f"from an unknown {trait}"
                
                store_traits[trait] = trait_value
            all_user_traits[user] = store_traits

        return all_user_traits

class PerspectivistLaMP():
    def __init__(self, model_identifier, persp_dataset, label, context,dataset_name):
        self.model_id = model_identifier
        self.label = label
        self.named = persp_dataset.named
        self.all_dataset = persp_dataset
        self.dataset = persp_dataset.name 
        self.context = context
        self.num_profiles = 5
        self.max_length = 1024
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.dataset_name = dataset_name
        if self.model_id == "mistralai/Mixtral-8x7B-Instruct-v0.1":
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.float16, device_map="auto")
            self.output_path = config.prediction_dir_mixtral
        elif self.model_id == "meta-llama/Meta-Llama-3.1-8B-Instruct":
            self.output_path = config.prediction_dir_llama
            self.model = pipeline(
                "text-generation",
                model=self.model_id,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device="cuda")
        else:
            print("LaMP requires mistralai/Mixtral-8x7B-Instruct-v0.1 or meta-llama/Meta-Llama-3.1-8B-Instruct")
            exit()
    
    def create_preprocessor(self):
        def preprocess(data):
            inputs = []
            targets = []
            for d in data:
                inputs.append(d["source"])
                targets.append(str(d["target"]))
            model_inputs = self.tokenizer(inputs, text_target=targets,return_tensors="pt", max_length=self.max_length, padding=True)
            return model_inputs
        return preprocess

    def to_hugging_dataset(dataset):
        def generator():
            for element in dataset:
                yield element
        return datasets.Dataset.from_generator(generator)
        
    def classification_query_corpus_maker(self, inp, profile):
        corpus = [f'{x["comment"]}' for x in profile]
        idx = inp.find('Input:')
        if idx == -1:
            return corpus, None
        query = inp[idx+len('Input:'):].strip()
        return corpus, query

    def create_prompt(self,input,profile,max_length,tokenizer,label):    
        inputs=input.split("Input:")
        if len(profile) == 0:
            print("No profiles found")
            return f'{inputs[0]}\nExample to label:\n {inputs[1]}\nYour output:'
        per_p_max_length = (max_length - 1 - 2 * (len(profile) - 1)) // len(profile)
        saved_tokens = 0
        prompts = []
        i = 1
        for p in profile:
            value_label = label if p[label] == 1 else "not " + label
            needed_part_len = len(tokenizer(f'Output: {value_label}')['input_ids'])
            tokens = tokenizer(p["comment"], max_length=per_p_max_length + saved_tokens - needed_part_len, truncation=True)
            saved_tokens += per_p_max_length - len(tokens['input_ids']) - needed_part_len
            new_text = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]
            new_text = new_text.replace('post :', "- Post:").replace(" reply :","\n- Reply:")
            prompt = f'Example {i}:\nInput:\n{new_text} \nOutput: {value_label}\n'
            i=i+1
            prompts.append(prompt)
        return f'{inputs[0]}\n{"".join(prompts)}\nExample to label:\n {inputs[1]}\nYour output:'

    def create_prompt_generator(self,num,label):
        tokenizer = AutoTokenizer.from_pretrained('facebook/contriever', padding=True)
        tokenizer.pad_token = tokenizer.eos_token
        def prompt(input,profile):
            selected_profs_cont = profile[:num]
            selected_profs = selected_profs_cont
            max_len_prompt = self.max_length - min(len(tokenizer(input)['input_ids']), int(0.6 * self.max_length))
            out = self.create_prompt(input,selected_profs,max_len_prompt,tokenizer,label)
            return out
        return prompt

    def evaluate_dataset(self):
        # Load the tokenizer and model
        self.tokenizer.pad_token = self.tokenizer.eos_token
        prompt_generator = self.create_prompt_generator(self.num_profiles, self.label)
        files = [file for file in os.listdir(config.data_lamp_dir) if file.lower().startswith(self.dataset.lower()) and "merged" in file and str(self.named) in file]
        if len(files) == 0:
            print("No merged files found. Please generate them using PrepareData.")
            exit()
        
        preprocessor = self.create_preprocessor()
        for file in files:
            with open(f'{config.data_lamp_dir}/{file}') as f:
                data = json.load(f)
                print(f'Processing {file}')
            if not os.path.exists(f'{config.data_lamp_dir}/output/'):
                os.makedirs(f'{config.data_lamp_dir}/output/')
            with open(f'{config.data_lamp_dir}/output/{file.replace("json","csv").replace("_merged","")}', 'w', newline='') as out_csv:
                writer = csv.writer(out_csv)
                field=["user_id","id","target","output"]
                writer.writerow(field)
                outputs = []
                with torch.no_grad():
                    for d in data:
                        print("Processing user: ", d)
                        for element in tqdm(data[d]):
                            profiles=[{
                                "id": element['id'],
                                "source": prompt_generator(element['input'], element['profile']),
                                "target": element[self.label]
                            }]
                            preprocessed_data = preprocessor(profiles)
                            inputs = {key: value.to("cuda") for key, value in preprocessed_data.items()}
                            for i in range(len(inputs["input_ids"])):
                                input_ids = inputs["input_ids"][i].unsqueeze(0)
                                attention_mask = inputs["attention_mask"][i].unsqueeze(0)
                                output = self.model.generate(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    max_new_tokens=100,
                                    do_sample=False
                                )
                                outputs.append(output)
                            generated_ids = output[:, inputs["input_ids"].shape[-1]:]
                            writer.writerow([d, element['id'],element[self.label],self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)])


    def merge_data(self, input, output, ranks,label):
        for data in input:
            for inp in input[data]:
                for id in output:
                    for o in output[id]:
                        if o['id'] == inp['id']:
                            outs = o[label]
                            break
                new_profile = []
                for x in ranks[data][inp['id']]:
                    for y in inp['profile']:
                        if y['id'] == x:
                            new_profile.append(y)
                            break
                inp['profile'] = new_profile
                inp[label] = outs
        return input

    def rank_profile(self,prompt):
        PrepareData(persp_dataset=self.all_dataset, dataset_config=prompt, named=self.named, context=self.context)
        contriver = AutoModel.from_pretrained("facebook/contriever").to("cuda:0")
        tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")

        contriver.eval()
        
        files = [file for file in os.listdir(config.data_lamp_dir) if file.lower().startswith(self.dataset.lower()) and "input" in file and str(self.named) in file]
        
        if len(files)==0:
            print("No input files found, you need to generate them with PrepareData")
            exit()
        
        for file in files:
            rank_dict={}
            with open(f'{config.data_lamp_dir}/{file}') as f:
                data = json.load(f)
                for user in tqdm(data):
                    rank_dict[user]={}
                    for comment in data[user]:
                        corpus,query = self.classification_query_corpus_maker(comment['input'],comment['profile'])
                        ranked_profile = self.retrieve_top_k_with_contriver(contriver, tokenizer, corpus, comment['profile'], query, len(comment['profile']), 16)
                        comment['profile'] = ranked_profile
                        rank_dict[user][comment['id']] = [x['id'] for x in ranked_profile]
            with open(f'{config.data_lamp_dir}/{file.replace("input","ranked")}', "w") as out:
                json.dump(rank_dict, out)
                
    def merge_profile(self):
        files = [file for file in os.listdir(config.data_lamp_dir) if file.lower().startswith(self.dataset.lower()) and "input" in file and str(self.named) in file]
        with open(f'{config.data_lamp_dir}/{self.dataset_name}_{self.named}_output.json') as output_file:
            out =json.load(output_file)
        if len(files)==0:
            print("No input files found, you need to generate them with Rank Profile")
            exit()
        for file in files:
            with open(f'{config.data_lamp_dir}/{file}') as f:
                data = json.load(f)
            if not os.path.isfile(f'{config.data_lamp_dir}/{file.replace("input","ranked")}'):
                print(f'File not found: {config.data_lamp_dir}/{file.replace("input","ranked")}')
                exit()
            with open(f'{config.data_lamp_dir}/{file.replace("input","ranked")}') as f:
                ranked_data = json.load(f)
            with open(f'{config.data_lamp_dir}/{file.replace("input","merged")}',"w") as merge_file:
                merged = self.merge_data(data,out, ranked_data,self.label)
                json.dump(merged,merge_file,indent=4)
                
    # LaMP rank_profiles methods
    def mean_pooling(self,token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings
    
    def batchify(self,lst, batch_size):
        return [lst[i:i+batch_size] for i in range(0, len(lst), batch_size)]

    def retrieve_top_k_with_contriver(self,contriver, tokenizer, corpus, profile, query, k, batch_size = 16):
        query_tokens = tokenizer([query], padding=True, truncation=True, return_tensors='pt').to("cuda:0")
        output_query = contriver(**query_tokens)
        output_query = self.mean_pooling(output_query.last_hidden_state, query_tokens['attention_mask'])
        scores = []
        batched_corpus = self.batchify(corpus, batch_size)
        for batch in batched_corpus:
            tokens_batch = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to("cuda:0")
            outputs_batch = contriver(**tokens_batch)
            outputs_batch = self.mean_pooling(outputs_batch.last_hidden_state, tokens_batch['attention_mask'])
            temp_scores = output_query.squeeze() @ outputs_batch.T
            scores.extend(temp_scores.tolist())
        _, topk_indices = torch.topk(torch.tensor(scores), k)
        return [profile[m] for m in topk_indices.tolist()]
        


