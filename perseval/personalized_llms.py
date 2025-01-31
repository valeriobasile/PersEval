import json
from tqdm import tqdm

from perseval.prompts import text_context_template, text_only_template


    

class PrepareData ():
    def __init__(self, persp_dataset, dataset_config, named=False, context=False):
        # self.training_split = persp_dataset.training_set
        self.adaptation_split = persp_dataset.adaptation_set        
        self.test_split = persp_dataset.test_set
        self.dataset_config = dataset_config
        self.dataset_name = self.dataset_config["dataset_name"]

        if context: 
            self.template = text_context_template
        else:
            self.template = text_only_template


        def generate_prompt (trait=None):
            format_args = {
                "txt_name": self.dataset_config["txt_name"],
                "data_source": self.dataset_config["data_source"],
                "task_adj": self.dataset_config["task_adj"],
                "labels": self.dataset_config["labels"],
            }

            # Determine the template to use
            if named and context:
                template_str = self.template["prompt_demographics"]
                format_args["trait"] = trait
                format_args["cntxt_name"] = self.dataset_config["cntxt_name"]
            elif named and not context:
                template_str = self.template["prompt_demographics"]
                format_args["trait"] = trait
            elif not named and context:
                template_str = self.template["prompt_zero"]
                format_args["cntxt_name"] = self.dataset_config["cntxt_name"]
            else:
                template_str = self.template["prompt_zero"]

            # Generate the prompt
            prompt = template_str.format(**format_args)
            return prompt
        



        def generate_input_data_named(store_prompts):
            trait_input_data = {}
            list_ids = []
            list_traits = []
            for user, traits_prompts in store_prompts.items():
                for trait, prompt in traits_prompts.items():
                    list_traits.append(trait)
                    if trait not in trait_input_data:
                        trait_input_data[trait] = {}  # Initialize for each trait if not already present
                    
                    # Store input data for the current user and trait
                    input_data = trait_input_data[trait]
                    
                    if user not in input_data:
                        input_data[user] = []

                    for key in self.test_split.annotation.keys():
                        u, t = key
                        
                        if u != user:  # Match the user in store_prompts with u
                            continue
                        
                        if t in self.test_split.texts:
                            text_data = self.test_split.texts[t]
                            
                            txt_name = self.dataset_config["txt_name"]
                            txt = text_data[txt_name]
                            if context: 
                                cntxt_name = self.dataset_config["cntxt_name"]
                                cntxt = text_data[cntxt_name]
                                formatted_text = (prompt + f" - {cntxt_name}: {cntxt} - {txt_name}: {txt}")
                            else:
                                formatted_text = (prompt + f" {txt_name}: {txt}")
                            list_ids.append(t)

                            input_data[u].append(
                                {
                                    "id": str(t),
                                    "input": formatted_text,
                                    "profile": profile.get(u, {})
                                }
                            )
                    
            # After collecting data for all users and traits, save the results for each trait
            for trait, input_data in trait_input_data.items():
                output_file = f"./data_LaMP/{self.dataset_name}_{trait}_input.json"
                with open(output_file, "w") as outfile:
                    json.dump(input_data, outfile, indent=4)
            
            print("instances test set: ",len(list_ids)//len(set(list_traits)))
            print("unique texts test set: ",len(set(list_ids)))
            print("number of traits: ", len(set(list_traits)))
            

        def generate_input_data_unnamed(prompt):
            input_data = {}
            list_ids = []

            for key in self.test_split.annotation.keys():
                u, t = key
                
                if u not in input_data:
                    input_data [u] = []

                if t in self.test_split.texts:
                    text_data = self.test_split.texts[t]
                    
                    txt_name = self.dataset_config["txt_name"]
                    txt = text_data[txt_name]
                    if context: 
                        cntxt_name = self.dataset_config["cntxt_name"]
                        cntxt = text_data[cntxt_name]
                        formatted_text = (prompt +f"- {cntxt_name}: {cntxt} - {txt_name}: {txt}")
                    else:
                        formatted_text = (prompt + f"{txt_name}: {txt}")
                    list_ids.append(t)

                    input_data[u].append(
                        {"id":str(t), 
                        "input":formatted_text, 
                        "profile":profile.get(u, {})
                        })
            print("instances test set: ",len(list_ids))
            print("unique texts test set: ",len(set(list_ids)))
            return input_data



        #create the profile based on the adapation set  
        profile = {}
        list_ids_a  = []
        for key,value in self.adaptation_split.annotation.items():
            u = key[0]
            t = key[1]
            if u not in profile:
                profile[u] = []

            if t in self.adaptation_split.texts:
                text_data = self.adaptation_split.texts[t]

                txt_name = self.dataset_config["txt_name"]
                txt = text_data[txt_name]
                if context: 
                    cntxt_name = self.dataset_config["cntxt_name"]
                    cntxt = text_data[cntxt_name]
                    formatted_text = f"{cntxt_name}: {cntxt} {txt_name}: {txt}"
                else:
                    formatted_text = f"{txt_name}: {txt}"
                list_ids_a.append(t)

            for k,v in value.items():
                profile[u].append({
                    "id":str(t),
                    "comment": formatted_text,
                    k:v
                })
        print("instances adaptation set: ",len(list_ids_a))
        print("unique texts adaptation set: ", len(set(list_ids_a)))


        # create the test set input 
        if named:
            store_prompts = {}
            for user, user_class in self.test_split.users.items():
                traits = user_class.traits
                store_traits = {}

                for trait,value in traits.items():
                    trait_value = None

                    for k,v in self.dataset_config["traits"].items():
                        if value[0] == k:
                            trait_value = v
                            break
                    if trait_value is None:
                        trait_value = "UNK" #--------------------------------------------> for the uknown values (e.g. Epic age; MHS gender)
                    
                    prompt = generate_prompt(trait=trait_value)

                    store_traits[trait] = prompt
                store_prompts [user] = store_traits
            generate_input_data_named(store_prompts)

        else:
            prompt = generate_prompt()
            input_data = generate_input_data_unnamed(prompt=prompt)
            
            with open(f"./data_LaMP/{self.dataset_name}_input.json", "w") as outfile: 
                json.dump(input_data, outfile)



        #create output file
        output_data = {}
        for key,value in self.test_split.annotation.items():
            u = key[0]
            t = key[1]
            if u not in output_data:
                output_data[u] = []

            for k,v in value.items():
                output_data[u].append({
                    "id":str(t),
                    k:v
                })
        with open(f"./data_LaMP/{self.dataset_name}_output.json", "w") as outfile: 
            json.dump(output_data, outfile)
