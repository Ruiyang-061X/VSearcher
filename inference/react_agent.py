import json5
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
from inference.prompt import *
import time
import traceback

from inference.tool.text_search import TextSearch
from inference.tool.image_search import ImageSearch
from inference.tool.visit import Visit

OBS_START = '<tool_response>'
OBS_END = '\n</tool_response>'

MAX_LLM_CALL_PER_RUN = 30

TOOL_CLASS = [
    TextSearch(),
    ImageSearch(),
    Visit(),
]
TOOL_MAP = {tool.name: tool for tool in TOOL_CLASS}


import random

from inference.inference_util import *


class MultiTurnReactAgent:

    def __init__(self, model, api_key, base_url):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.generate_cfg = {
            "temperature": 0.6,
            "top_p": 0.95,
            "presence_penalty": 1.5,
        }

    def call_server(self, msgs, max_tries=5):
        openai_api_key = self.api_key
        openai_api_base = self.base_url

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            timeout=600.0,
        )

        base_sleep_time = 1 
        for attempt in range(max_tries):
            try:
                print(f"--- Attempting to call the service, try {attempt + 1}/{max_tries} ---")
                chat_response = client.chat.completions.create(
                    model=self.model,
                    messages=msgs,
                    stop=["\n<tool_response>", "<tool_response>"],
                    temperature=self.generate_cfg.get('temperature', 0.6),
                    top_p=self.generate_cfg.get('top_p', 0.95),
                    logprobs=True,
                    max_tokens=10000,
                    presence_penalty=self.generate_cfg.get('presence_penalty', 1.1),
                )
                content = chat_response.choices[0].message.content

                # OpenRouter provides API calling. If you want to use OpenRouter, you need to uncomment line 89 - 90.
                # reasoning_content = "<think>\n" + chat_response.choices[0].message.reasoning.strip() + "\n</think>"
                # content = reasoning_content + content                

                if content and content.strip():
                    print("--- Service call successful, received a valid response ---")
                    return content.strip()
                else:
                    print(f"Warning: Attempt {attempt + 1} received an empty response.")

            except (APIError, APIConnectionError, APITimeoutError) as e:
                print(f"Error: Attempt {attempt + 1} failed with an API or network error: {e}")
                traceback.print_exc()
            except Exception as e:
                print(f"Error: Attempt {attempt + 1} failed with an unexpected error: {e}")

            if attempt < max_tries - 1:
                sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)
                sleep_time = min(sleep_time, 30) 
                
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                print("Error: All retry attempts have been exhausted. The call has failed.")
        
        return f"vllm server error!!!"

    def _run(self, data):
        start_time = time.time()
        question = data['question']
        image_url = data['image_path']
        answer = data['answer']
        print(f"question: {question}")
        print(f"image_url: {image_url}")
        print(f"answer: {answer}")
        system_prompt = SYSTEM_PROMPT
        cur_date = today_date()
        system_prompt = system_prompt + str(cur_date)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}}, {"type": "text", "text": question}]}]
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round = 0
        while num_llm_calls_available > 0:
            # Check whether time is reached
            if time.time() - start_time > 150 * 60:  # 150 minutes in seconds
                prediction = 'No answer found after 2h30mins'
                termination = 'No answer found after 2h30mins'
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination
                }
                return result
            round += 1
            num_llm_calls_available -= 1
            content = self.call_server(messages)
            if '<tool_response>' in content:
                pos = content.find('<tool_response>')
                content = content[:pos]
            if '<tool_call>' in content and '</tool_call>' in content:
                content = content.split('</tool_call>')[0] + "</tool_call>"
            print(f'Round {round}: {content}')
            messages.append({"role": "assistant", "content": content.strip()})
            if content == "vllm server error!!!":
                prediction = 'vllm server error!!!'
                termination = 'vllm server error!!!'
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination
                }
                return result
            if '<tool_call>' in content and '</tool_call>' in content:
                tool_call = content.split('<tool_call>')[1].split('</tool_call>')[0]
                try:
                        tool_call = clean_tool_call(tool_call)
                        tool_call = json5.loads(tool_call)
                        tool_name = tool_call.get('name', '')
                        tool_args = tool_call.get('arguments', {})
                        result = self.custom_call_tool(tool_name, tool_args, data)
                        print(result)
                        if tool_name != "image_search":
                            result = "<tool_response>\n" + result + "\n</tool_response>"
                            messages.append({"role": "user", "content": result})
                        else:
                            search_results, search_results_image = result
                            result = "<tool_response>\n" + search_results + "\n</tool_response>"
                            messages.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}} for image_url in search_results_image] + [{"type": "text", "text": result}]})
                except Exception as e:
                    result = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field.'
                    traceback.print_exc()
            if '<answer>' in content and '</answer>' in content:
                termination = 'answer'
                break
            if num_llm_calls_available <= 0 and '<answer>' not in content:
                messages[-1]['content'] = 'Sorry, the number of llm calls exceeds the limit.'

        if '<answer>' in messages[-1]['content']:
            prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
            termination = 'answer'
        else:
            prediction = 'No answer found.'
            termination = 'answer not found'
            if num_llm_calls_available == 0:
                termination = 'exceed available llm calls'
        result = {
            "question": question,
            "image_url": image_url,
            "answer": answer,
            "messages": messages,
            "prediction": prediction,
            "termination": termination
        }
        return result

    def custom_call_tool(self, tool_name, tool_args, data):
        if tool_name in TOOL_MAP:
            tool_args["params"] = tool_args
            if tool_name == "image_search":
                tool_args = {
                    "image_url": data["image_path"]
                }
                result = TOOL_MAP['image_search'].call(tool_args)
            else:
                result = TOOL_MAP[tool_name].call(tool_args)
            return result
        else:
            return f"Error: Tool {tool_name} not found"


if __name__ == "__main__":
    agent = MultiTurnReactAgent(
        model="Ruiyang-061X/VSearcher-8B",
        api_key="EMPTY",
        base_url="http://127.0.0.1:9000/v1"
    )

    # replace with your own data
    # "image_path" is web url of image
    data = {
        "question": "",
        "image_path": "",
        "answer": ""
    }
    
    result = agent._run(data)
    print(result)