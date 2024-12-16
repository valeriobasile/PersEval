from perseval.data import *
from perseval.model import *
from perseval.evaluation import *
from transformers.utils import logging
import argparse
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
"""


# options for label:
# EPIC   -> ["irony"]
# BREXIT -> ["hs", "offensiveness", "aggressiveness", "stereotype"]
# DICES  -> ["degree_of_harm"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-name", 
        type=str, 
        required=False,
        default="Epic", 
        choices=['Epic', 'Brexit', 'DICES'],
        help="Name of the dataset to use. Options: ['Epic', 'Brexit', 'DICES']")
    parser.add_argument(
        "--model-name",
        type=str,
        required=False,
        default="roberta-base",
        help="Name of the transformer model to run inference on")
    parser.add_argument(
        "--label",
        type=str,
        required=False,
        default="irony")
    parser.add_argument(
        "--use-llm",
        action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    # initialize the dataset
    if args.dataset_name == "Epic":
        perspectivist_dataset = Epic()
    elif args.dataset_name == "Brexit":
        perspectivist_dataset = Brexit()
    elif args.dataset_name == "DICES":
        perspectivist_dataset = DICES()
    perspectivist_dataset.get_splits(user_adaptation="train", extended=False, named=True)

    # create the model
    if args.use_llm:
        pass
        #model = PerspectiveLLM(args.model_name, 
                                #perspectivist_dataset, 
                                #label=args.label)
    else:
        model = PerspectivistEncoder(args.model_name, 
                                perspectivist_dataset, 
                                label=args.label)

    # train/test the model if it is not an LLM, test only if it is an LLM
    if not args.use_llm:
        trainer = model.train()
        model.predict(trainer)  # <-- Predictions are saved in the "predictions" folder, 
    else:                       #     The file must contain three columns:
        #model.predict()         #     "user_id", "text_id", "label"
        pass

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


if __name__ == "__main__":
    main()