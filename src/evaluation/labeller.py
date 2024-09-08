import logging
import pathlib
import time
from dotenv import load_dotenv
import os
from qa_metrics.prompt_llm import CloseLLM

def load_environment_variables():
    path_env = pathlib.Path("/export/usuarios_ml4ds/lbartolome/Repos/repos_con_carlos/RAG_tool/.env")
    print(path_env)
    load_dotenv(path_env)
    api_key = os.getenv("OPENAI_API_KEY")
    return api_key

def initialize_gpt_model(api_key):
    gpt_model = CloseLLM()
    gpt_model.set_openai_api_key(api_key)
    return gpt_model

class TopicLabeller(object):
    def __init__(self, prompt_template="src/evaluation/prompt_labeller.txt", gpt_model='gpt-3.5-turbo', temperature =0, max_tokens = 500, logger=None):
        
        self._logger = logger if logger else logging.getLogger(__name__)
        self.model_engine = gpt_model
        api_key = load_environment_variables()
        gpt_model = initialize_gpt_model(api_key)
        self.prompt_template = self._load_prompt_template(prompt_template)
        self.gpt_model = gpt_model
        
        self.params = {
            'temperature': temperature,
            'max_tokens': max_tokens
        }
        return
    
    def _load_prompt_template(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            file_contents = file.read()
        return file_contents
    
    def label_topics(self, topic_keys, path_save):
        
        tpc_labels = []
        for tpc in topic_keys:
            this_tpc_promt = self.prompt_template.format(tpc)
            print(f"Topic: {tpc}")
            llm_response = self.gpt_model.prompt_gpt(
                prompt=this_tpc_promt, model_engine=self.model_engine, **self.params
            )
            time.sleep(1)
            tpc_labels.append(llm_response)
            print(f"Label: {llm_response}")
            
        with open(path_save, 'w', encoding='utf-8') as file:
            for label in tpc_labels:
                file.write(label)
                
        return tpc_labels