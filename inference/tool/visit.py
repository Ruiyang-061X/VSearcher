import requests

from inference.Qwen2_5 import Qwen2_5
from inference.prompt import SUMMARY_PROMPT
from inference.inference_util import *

import os
JINA_KEY = os.environ.get("JINA_KEY")


class Visit:

    name = "visit"

    def __init__(self):
        self.key = JINA_KEY
        self.summary_model = Qwen2_5()
        self.cache = load_json(".cache/visit.json")

    def call(self, params):
        if params["url"] in self.cache:
            page_content = self.cache[params["url"]]
        else:
            url = "https://r.jina.ai/" + params["url"]
            headers = {
                "Authorization": f"Bearer {self.key}"
            }
            response = requests.get(url, headers=headers)
            page_content = response.text.split("Markdown Content:")[1]
            self.cache[params["url"]] = page_content
            if len(self.cache) % 10 == 0:
                save_json(self.cache, ".cache/visit.json")
        prompt = SUMMARY_PROMPT
        prompt = prompt.replace("[WEB PAGE CONTENT]", page_content)
        prompt = prompt.replace("[USER GOAL]", params["goal"])
        summary = self.summary_model.query(prompt)
        return summary


if __name__ == "__main__":
    visit = Visit()
    params = {
        "url": "https://en.wikipedia.org/wiki/Shenzhen",
        "goal": "summarize information is this page"
    }
    summary = visit.call(params)
    print(summary)