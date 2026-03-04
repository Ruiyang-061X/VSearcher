from openai import OpenAI
from transformers import AutoTokenizer


class Qwen2_5:

    def __init__(self):
        openai_api_key = "EMPTY"
        openai_api_base = "http://127.0.0.1:8000/v1"

        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-72B-Instruct")

    def query(self, prompt):
        tokens = self.tokenizer.encode(prompt)
        if len(tokens) > 32768:
            prompt = self.tokenizer.decode(tokens[:32000])

        chat_response = self.client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return chat_response.choices[0].message.content

if __name__ == "__main__":
    qwen2_5 = Qwen2_5()
    prompt = "hello"
    response = qwen2_5.query(prompt)
    print(response)