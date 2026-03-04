import requests
from inference.inference_util import *

import os
SERPER_KEY = os.environ.get("SERPER_KEY")


class TextSearch:

    name = "text_search"

    def __init__(self):
        self.cache = load_json(".cache/text_search.json")
        self.top_k = 5

    def call(self, params):
        if params["query"] in self.cache:
            search_results = self.cache[params["query"]]
        else:
            url = "https://google.serper.dev/search"

            payload = {
                "q": params["query"]
            }

            headers = {
                'X-API-KEY': SERPER_KEY,
                'Content-Type': 'application/json'
            }

            response = requests.request("POST", url, headers=headers, json=payload)
            response = response.json()

            items = response["organic"]
            if len(items) > self.top_k:
                items = items[:self.top_k]

            search_results = ""
            for idx, item in enumerate(items):
                search_results += f"{idx + 1}. [Link] {item['link']} [Title] {item['title']} [Snippet] {item['snippet']}\n"
            self.cache[params["query"]] = search_results
            if len(self.cache) % 10 == 0:
                save_json(self.cache, ".cache/text_search.json")
        return search_results


if __name__ == "__main__":
    text_search = TextSearch()
    query = "UEFA Euro 2024 Final Schedule"
    params = {
        "query": query
    }
    search_results = text_search.call(params)
    print(search_results)
