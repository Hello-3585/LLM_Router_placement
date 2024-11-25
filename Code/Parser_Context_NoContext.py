import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)y
import random
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
HF_token ="Your HF token"

#LOAD DATA
dataset_json_path = ""
dataset_list = []

# Iterate through files in the Dataset_json folder
for file_name in os.listdir(dataset_json_path):
    file_path = os.path.join(dataset_json_path, file_name)
    
    # Check if the file is a JSON file
    if file_name.endswith(".json"):
        try:
            # Read and load JSON content
            with open(file_path, 'r') as file:
                data = json.load(file)
                dataset_list.append(data)
        except Exception as e:
            print(f"Error reading {file_name}: {e}")

dataset_json_path = "..../Environment"
environment_list = []

# Iterate through files in the Dataset_json folder
for file_name in os.listdir(dataset_json_path):
    file_path = os.path.join(dataset_json_path, file_name)
    
    with open(file_path, 'r') as file:
        data = file.read()
    file.close()
    environment_list.append(data)
    
dataset_json_path = "...../Routers"
router_list = []

# Iterate through files in the Dataset_json folder
for file_name in os.listdir(dataset_json_path):
    file_path = os.path.join(dataset_json_path, file_name)
    
    with open(file_path, 'r') as file:
        data = file.read()
    file.close()
    router_list.append(data)

with open("..../Defaults.txt",'r') as file:
    defaults=file.read()
file.close()

class Parser_Agent:
    def __init__(self):
        self.llama_model =None
        self.tokenizer = None
    def config(self):
        use_8bit = True
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=use_8bit,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )

        # Load the Llama 3 8B model and tokenizer
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"  
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llama_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                   quantization_config=bnb_config,
                                                   device_map="auto",
                                                   token=HF_token)
    def generate(self,request):
        system={"role": "system", 
            "content": 
        """
        You need to provide a jsonised output containing the following sections extracted from the query:
        {
        "Parsed_Parameters": {
                        "X_dimension": <float>,
                        "Y_dimension": <float>,
                        "Number_of_Users": <int>,
                        "Number_of_Routers": <int>,
                        "X_User_Distribution": <string>,
                        "Y_User_Distribution": <string>,
                        "Height": <float>,
                        "K": <float>,
                        "Alpha": <float>,
                        "Transmit_Power": <float>,
                        "User Equipment Height":<float>
                    }
        }

        Example of what your output should look like:
        {
        "Parsed_Parameters": {
            "X_dimension": 30.0,
            "Y_dimension": 40.0,
            "Number_of_Users": 100,
            "Number_of_Routers": 6,
            "X_User_Distribution": "Exponential",
            "Y_User_Distribution": "Exponential",
            "Height": 4.0,
            "K": 15.0,
            "Alpha": 0.005,
            "Transmit_Power": 19,
            "User Equipment Height": 1.0
        }
        }

        Any information missing in the user request must be retreived from router specification files or environment files in the context files provided.

        X and Y Distribution must be one of Uniform, Gaussian, Exponential, Bi-Exponential

        Only provide the output parsed json without any reasoning or any other statements.

        Extract only the parameters mentioned in the example output.
        """
        }
        context='CONTEXT: \n'
        for f in environment_list:
            context=context+f
        for f in router_list:
            context=context+f
        context=context+defaults
        self.config()
        message=[system,
    {"role":"user","content":context+"\n User Request: \n " + request}]
        input_ids = self.tokenizer.apply_chat_template(message, return_tensors="pt",padding=True, truncation=True,).to(self.llama_model.device)
        with torch.no_grad():
            output = self.llama_model.generate(input_ids, max_length=input_ids.shape[1]+500, num_return_sequences=1, temperature=0.1);
        input_length = input_ids.shape[1] 
        clean_prompt=output[0][input_length+3:];
        answer=self.tokenizer.decode(clean_prompt, skip_special_tokens=True);
        return answer