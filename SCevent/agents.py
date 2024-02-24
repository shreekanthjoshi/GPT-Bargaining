import openai
import os
import time
import re

from copy import deepcopy
from pprint import pprint
from tenacity import retry, stop_after_attempt, wait_chain, wait_fixed

STOP_AFTER_ATTEMPT=4

def initial_instructions(instructions):
    """Load initial instructions from textual format to a python dict"""
    pattern = r"==== (SYSTEM|USER|ASSISTANT) ===="

    # Use re.split to split the string by the pattern
    with open(instructions) as f:
        content = f.read()
        content = re.split(pattern, content)
        content_ = []
        for c in content: 
            if(c != ""): content_.append(c)
        content = content_
        l = len(content)
        assert(l % 2 == 0)
        initial_instruction = []
        for i in range(0, l, 2):
            instruction = {"role": content[i].strip().lower().replace("====", "").replace(" ", "").strip(), 
                           "content": content[i+1].strip()
                           }
            initial_instruction.append(instruction)
    return initial_instruction

def wprint(s, fd=None, verbose=False):
    if(fd is not None): fd.write(s + '\n')
    if(verbose): print(s)
    return 

@retry(stop=stop_after_attempt(STOP_AFTER_ATTEMPT), 
        wait=wait_chain(*[wait_fixed(3) for i in range(2)] +
                       [wait_fixed(5) for i in range(1)]))
def completion_with_backoff(model, messages):
    """OpenAI API wrapper, if network error then retry 3 times"""

    openai.api_type = "azure"
    openai.api_base = "https://tfccgpoc.openai.azure.com/"
    openai.api_version = "2023-09-15-preview"
    openai.api_key = os.getenv("OPENAI_API_KEY")
    return openai.ChatCompletion.create(
        engine="TFccgUsecase",
        messages = messages,
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
        )



class SCAgent(object):
    """GPT Agent base class 
     
    """
    def __init__(self, 
                 initial_dialog_history=None,
                 agent_type="", # "counselor", "patient",  "moderator"
                 system_instruction="You are a helpful AI assistant", 
                 engine="gpt-3.5-turbo",
                 api_key="",
                 
                ):
        """Initialize the agent"""
        super().__init__()
        
        self.agent_type = agent_type
        self.engine = engine
        self.api_key = api_key
        

        if(initial_dialog_history is None):
            self.dialog_history = [{"role": "system", "content": system_instruction}]
        else:
            self.initial_dialog_history = deepcopy(initial_dialog_history)
            self.dialog_history = deepcopy(initial_dialog_history)

        self.last_prompt = ""
        return 
    
    def reset(self):
        """Reset dialog history"""
        self.dialog_history = deepcopy(self.initial_dialog_history)
        return 

    def call_engine(self, messages):
        
            try:
                response = completion_with_backoff(
                        model=self.engine,
                         messages=messages
                      )
                message = response['choices'][0]['message']  
                assert(message['role'] == 'assistant')
            except:         
                print('timeout error')
                time.sleep(5)
                message={}
                message['content']=''

            return message
        
    def call(self, prompt):
        """Call the agent with a prompt.  
        """
         
        prompt = {"role": "user", "content": prompt}
        self.dialog_history.append(prompt)
        self.last_prompt = prompt['content']
        
        messages = list(self.dialog_history)
        # messages.append(prompt)
        
        message = self.call_engine(messages)
        if not message:
            print('message empty')
        else:
            self.dialog_history.append(dict(message))

        return message['content']

    @property
    def last_response(self):
        return self.dialog_history[-1]['content']
    
    @property
    def history(self):
        for h in self.dialog_history:
            print('%s:  %s' % (h["role"], h["content"]))
        return 
    
class PatientAgent(SCAgent):

    def __init__(self, 
                 initial_dialog_history=None,
                 agent_type="buyer",
                 engine="gpt-3.5-turbo",
                 api_key="",
                 patient_instruction="patient",
                ):
        """Initialize the patient agent"""
        super().__init__(initial_dialog_history=initial_dialog_history, 
                         agent_type=agent_type, 
                         engine=engine,
                         api_key=api_key,
                         )
        self.patient_instruction = patient_instruction

    
    def reset(self):
        """Reset dialog history"""
        self.dialog_history = deepcopy(self.initial_dialog_history)


class CounselorAgent(SCAgent):
    
    def __init__(self, 
                 initial_dialog_history=None,
                 agent_type="counselor",
                 engine="gpt-3.5-turbo",
                 api_key="",
                ):
        """Initialize the counselor agent"""
        super().__init__(initial_dialog_history=initial_dialog_history, 
                         agent_type=agent_type, 
                         engine=engine,
                         api_key=api_key,
                         )

    
    def reset(self):
        """Reset dialog history"""
        self.dialog_history = deepcopy(self.initial_dialog_history)


class ModeratorAgent(SCAgent):
    
    def __init__(self, 
                 initial_dialog_history=None,
                 agent_type="moderator",
                 engine="gpt-3.5-turbo",
                 api_key="",
                 trace_n_history=2,
                ):
        """Initialize the moderator agent"""
        super().__init__(initial_dialog_history=initial_dialog_history, 
                         agent_type=agent_type, 
                         engine=engine,
                         api_key=api_key
                         )

        self.trace_n_history = trace_n_history
        return
    
    def moderate(self, 
                 dialog_history, who_was_last 
                 ):
        """Moderate the conversation between the counselor and the patient"""
        history_len = len(dialog_history)
        if(who_was_last == "patient"):
            prompt = "patient: %s\n" % dialog_history[history_len - 1]["content"]
            offset = 1
        else: 
            prompt = "counselor: %s\n" % dialog_history[history_len - 1]["content"]
            offset = 0

        for i in range(self.trace_n_history - 1):
            idx = history_len - i - 2
            content = dialog_history[idx]["content"]
            if(i % 2 == offset):
                prompt = "patient: %s\n" % content + prompt
            else:
                prompt = "counselor: %s\n" % content + prompt
        
        prompt += "question: has the conversation between counsellor and patient reached a conclusion Yes or No\nanswer:"
        self.last_prompt = prompt
        
        messages = deepcopy(self.dialog_history)
        messages[-1]['content'] += "\n\n" + prompt

        response = self.call_engine(messages)
        return response['content']
    