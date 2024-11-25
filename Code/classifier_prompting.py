import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

HF_token ="Your HF token"

class Classify_Prompting:
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
        """You are an expert in telecommunication systems. Your task is to classify the user request to one of the five optimization classes.

        Class names and Objectives are as follows:
        1. KM (K-Means): Distance Reduction
        2. WKHM (Weighted K-Harmonic Means): Balancing Load and Power Loss Optimization
        3. CKM (Constrained K-Means): Load Balancing 
        4. KC (K-Centers): Maximize Minimum RSRP[Received Signal Reference Power]
        5. KHM (K-Harmonic Means): Power Loss Optimization/ Maximising Received Signal Reference Power

        IMPORTANT RULES:
        1. For the final answer provide ONLY the class number within <>. For example <1>,<2>,<3>,<4>,<5>.
        2. DO NOT RETURN ANYTHING ELSE OTHER THAN THE CLASS NUMBER WITHIN "<>".
        3. RETURNING ANYTHING ELSE WOULD RESULT IN A PENALTY.
        4. LIMIT RESPONSE TO A MAXIMUM OF 10 CHARACTERS
        5. REMEMBER THERE ARE ONLY 5 CLASS  NUMBERS : 1 , 2, 3, 4, 5 . DON"T GENERATE ANY OTHER CLASS NUMBER.
        """
        }
        self.config()
        message=[system,
            {"role":"user","content":request}]
        input_ids = self.tokenizer.apply_chat_template(message, return_tensors="pt",padding=True, truncation=True,).to(self.llama_model.device)
            
        with torch.no_grad():
            output = self.llama_model.generate(input_ids, max_length=input_ids.shape[1]+150, num_return_sequences=1, temperature=1);
            
        modified_prompt = self.tokenizer.decode(output[0], skip_special_tokens=True)
        answer = modified_prompt.split("assistant")[-1].strip()
        return answer
