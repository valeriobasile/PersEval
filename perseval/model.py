import sys
sys.path.append("..")

import pandas as pd
import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    Trainer, set_seed, TrainingArguments 
from datasets import Dataset
import torch
from sklearn.utils import compute_class_weight
import numpy as np

from . import config

class PerspectivistEncoder():
    def __init__(self, model_identifier, training_split, test_split, label, traits, named):
        self.model_id = model_identifier
        self.training_split = training_split
        self.test_split = test_split
        self.label = label
        self.traits = traits
        self.named = named
    
        self.tokenizer = AutoTokenizer.from_pretrained(model_identifier)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_identifier)


    def train(self):
        set_seed(config.seed)
        data = {"train" : self.__generate_data(self.training_split)[0]}   
        # computer class weight (in case labels are unbalanced)        
        try:
            class_weights = compute_class_weight(
                "balanced",
                classes=[0, 1],
                y=data["train"]["labels"].float().numpy()).astype("float32")
        except:
            print("unable to balance classes")
            class_weights = np.array([1, 1]).astype("float32")


        print('We will use the device:', torch.cuda.get_device_name(0))
        training_args = TrainingArguments(
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
        with open(config.prediction_dir+"/outputs.csv", "w") as fo:
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
        self.__add_special_tokens_to_tokenizer(split, self.named)
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
    
    
    def __add_special_tokens_to_tokenizer(self, split, named):
        special_tokens_dict = {"additional_special_tokens": []}
        new_tokens = set()
        for user in split.users:
            new_tokens.add('<{}>'.format(user))
            if named:
                for dim in self.traits:
                    for trait in list(self.traits[dim]):
                        new_tokens.add('<{}:{}>'.format(dim, trait))
        special_tokens_dict['additional_special_tokens'] = list(new_tokens)        
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer)) 


class CustomTrainer(Trainer):
    """ Custom Trainer class to implement a custom looss function
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