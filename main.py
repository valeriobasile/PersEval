from perseval.data import *
from perseval.model import *
from perseval.evaluation import *
from transformers.utils import logging
logging.set_verbosity_error() 

"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

prompt = "<s> [INST] hello. how are you? [/INST]"
model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
output = tokenizer.batch_decode(generated_ids)[0]
print(output)
exit()
"""
from transformers import pipeline
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

messages = [
    {"role": "user", "content": "Hello. How are you? reply within brackets like this {}"},
]
outputs = pipe(
    messages,
    max_new_tokens=256,
    do_sample=False,
)
assistant_response = outputs[0]["generated_text"][-1]["content"]
print(outputs)
exit()



perspectivist_dataset = DICES()
perspectivist_dataset.get_splits(user_adaptation="train", extended=False, named=True)

# options for label:
# EPIC   -> ["irony"]
# BREXIT -> ["hs", "offensiveness", "aggressiveness", "stereotype"]
# DICES  -> ["degree_of_harm"]
model = PerspectivistEncoder("roberta-base", 
                            perspectivist_dataset, 
                            label="degree_of_harm")

trainer = model.train()
model.predict(trainer) # <-- Predictions are saved in the "predictions" folder, 
                       #     The file must contain three columns:
                       #     "user_id", "text_id", "label"


evaluator = Evaluator(prediction_path="predictions/predictions_%s_%s_%s_%s.csv" % (perspectivist_dataset.name, perspectivist_dataset.named, perspectivist_dataset.user_adaptation, perspectivist_dataset.extended),
                      test_set=perspectivist_dataset.test_set,
                      label="degree_of_harm")
evaluator.global_metrics()
evaluator.annotator_level_metrics()
evaluator.text_level_metrics()
evaluator.trait_level_metrics()

# You can also access the metrics from
#print(evaluator.global_metrics_dic)
#print(evaluator.annotator_level_metrics_dic)
#print(evaluator.annotator_based_macro_avg)
#print(evaluator.text_level_metrics_dic)
#print(evaluator.text_based_macro_avg)
#print(evaluator.trait_level_metrics_dic)
#print(evaluator.trait_based_macro_avg)
