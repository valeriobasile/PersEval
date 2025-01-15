import json
from perseval.prompts import text_context_template, text_only_template
from perseval.prompts import EPIC, BREXIT, MHS, DICES, MD_Agreement


    

class PrepareData ():
    def __init__(self, persp_dataset, named=False, context=False):
        # self.training_split = persp_dataset.training_set
        self.adaptation_split = persp_dataset.adaptation_set        
        self.test_split = persp_dataset.test_set
        self.dataset = EPIC  ##################################################---->CHANGE THIS 

        if context: 
            self.template = text_context_template
        else:
            self.template = text_only_template

        def generate_prompt (trait = None):
            format_args = {
                "txt_name": self.dataset["txt_name"],
                "data_source": self.dataset["data_source"],
                "task_adj": self.dataset["task_adj"],
                "labels": self.dataset["labels"],
            }

            # Determine the template to use
            if named and context:
                template_str = self.template["prompt_demographics"]
                format_args["trait"] = trait
                format_args["cntxt_name"] = self.dataset["cntxt_name"]
            elif named and not context:
                template_str = self.template["prompt_demographics"]
                format_args["traits"] = trait
            elif not named and context:
                template_str = self.template["prompt_zero"]
                format_args["cntxt_name"] = self.dataset["cntxt_name"]
            else:
                template_str = self.template["prompt_zero"]

            # Generate the prompt
            prompt = template_str.format(**format_args)
            return prompt
        


        #create the profile based on the adapation set  
        profile = {}
        for key,value in self.adaptation_split.annotation.items():
            u = key[0]
            t = key[1]
            if u not in profile:
                profile[u] = []

            if t in self.adaptation_split.texts:
                text_data = self.adaptation_split.texts[t]

                txt_name = self.dataset["txt_name"]
                txt = text_data[txt_name]
                if context: 
                    cntxt_name = self.dataset["cntxt_name"]
                    cntxt = text_data[cntxt_name]
                    formatted_text = f"{cntxt_name}: {cntxt} {txt_name}: {txt}"
                else:
                    formatted_text = f"{txt_name}: {txt}"

            for k,v in value.items():
                profile[u].append({
                    "id":str(t),
                    "comment": formatted_text,
                    k:v
                })



        # create the test set input 
        input_data = {}

        for key in self.test_split.annotation.keys():
            u, t = key
            
            if u not in input_data:
                input_data [u] = []

            if t in self.test_split.texts:
                text_data = self.test_split.texts[t]
                
                txt_name = self.dataset["txt_name"]
                txt = text_data[txt_name]
                if context: 
                    cntxt_name = self.dataset["cntxt_name"]
                    cntxt = text_data[cntxt_name]
                    formatted_text = f"{cntxt_name}: {cntxt} {txt_name}: {txt}"
                else:
                    formatted_text = f"{txt_name}: {txt}"


                input_data[u].append(
                    {"id":str(t), 
                     "input":formatted_text, 
                     "profile":profile.get(u, {})
                     })
            else:
                input_data[u].append(
                    {"id":str(t),
                    "profile":profile.get(u, {})
                    })
        
        with open("input.json", "w") as outfile: 
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
        # with open("output.json", "w") as outfile: 
        #     json.dump(output_data, outfile)

    





