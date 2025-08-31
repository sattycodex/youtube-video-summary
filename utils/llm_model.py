from langchain_core.language_models.llms import LLM, LLMResult, Generation
import requests
import os
from dotenv import load_dotenv

load_dotenv()

EURI_API_KEY = os.getenv("EURI_API_KEY")

def generate_completion(message,temperature=0.7,max_tokens=500):
    url = "https://api.euron.one/api/v1/euri/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EURI_API_KEY}"
    }
    payload = {
        "messages": [
            {
                "role": "user",
                "content": message
            }
        ],
        "model": "gpt-4.1-nano",
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    response = requests.post(url, headers=headers, json=payload)
    data = response.json()
    return data['choices'][0]['message']['content']

        
class EuriLLM(LLM):
    def _call(self, prompt: str, stop=None, **kwargs) -> str:
        # Compose the messages as expected by generate_completion
        messages = [
            {"role": "system", "content": "You are a helpful assistant that helps people find information."},
            {"role": "user", "content": prompt}
        ]
        # Join messages for the API
        # Since generate_completion expects a string, we join the contents
        message_content = " ".join([m["content"] for m in messages])
        return generate_completion(message_content)
    
    def _generate(self, prompts, stop = None, **kwargs)->LLMResult:
        generations=[]
        for prompt in prompts:
            output=self._call(prompt)
            generations.append([Generation(text=output)])
        return LLMResult(generations=generations)
    
    def with_structured_output(self, schema):
        # Return a wrapper that parses the output according to the schema
        def structured_call(prompt):
            result = self._call(prompt)
            # Simple parsing: expects "name: satyam, age: 20"
            output = {}
            for key in schema:
                # Find "key: value" in result
                import re
                match = re.search(rf"{key}\s*:\s*([^\s,]+)", result, re.IGNORECASE)
                if match:
                    value = match.group(1)
                    # Cast to expected type
                    output[key] = schema[key](value)
            return output
        # Return an object with a _call method
        class Wrapper:
            def _call(self, prompt):
                return structured_call(prompt)
        return Wrapper()
    
    @property
    def _identifying_params(self):
        return {"model": "gpt-4.1-nano"}    
    
    @property
    def _llm_type(self) -> str:
        return "euri"