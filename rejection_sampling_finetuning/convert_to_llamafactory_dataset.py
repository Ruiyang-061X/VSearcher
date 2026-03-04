import argparse

from rejection_sampling_finetuning.rft_util import *


def convert(trajectory_list):
    llamafactory_dataset = []

    for trajectory in trajectory_list:
        llamafactory_data = {}
        llamafactory_data["messages"] = []
        llamafactory_data["images"] = []
        for message in trajectory["messages"]:
            if message["role"] == "system":
                llamafactory_data["messages"].append(message)
            elif message["role"] == "user":
                if isinstance(message["content"], str):
                    llamafactory_data["messages"].append(message)
                    continue
                current_message = {
                    "role": "user",
                    "content": ""
                }
                current_images = []
                for content in message["content"]:
                    if content["type"] == "image_url":
                        current_images.append(content["image_url"]["url"])
                    elif content["type"] == "text":
                        current_message["content"] = content["text"]
                current_message["content"] = "<image>" * len(current_images) + current_message["content"]
                llamafactory_data["messages"].append(current_message)
                llamafactory_data["images"].extend(current_images)
            elif message["role"] == "assistant":
                llamafactory_data["messages"].append(message)
        llamafactory_dataset.append(llamafactory_data)

    return llamafactory_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory_jsonl_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    
    args = parser.parse_args()

    trajectory_list = load_jsonl(args.trajectory_jsonl_path)
    llamafactory_dataset = convert(trajectory_list)
    save_json(llamafactory_dataset, args.output_path, indent=2)