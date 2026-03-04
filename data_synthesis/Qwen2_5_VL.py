from openai import OpenAI


class Qwen2_5_VL:

    def __init__(self):
        openai_api_key = "EMPTY"
        openai_api_base = "http://127.0.0.1:8001/v1"

        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

    def query(self, question, image_url):
        chat_response = self.client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-72B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            },
                        },
                    ]
                }
            ]
        )
        return chat_response.choices[0].message.content

if __name__ == "__main__":
    qwen2_5_vl = Qwen2_5_VL()
    question = "Describe the image in detail."
    image_url = "https://raw.githubusercontent.com/image-storage-rl/image_storage/main/mmsearch/mmsearch_51.jpg"
    response = qwen2_5_vl.query(question, image_url)
    print(response)