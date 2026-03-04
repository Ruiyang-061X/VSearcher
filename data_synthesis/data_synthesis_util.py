import json


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, file_path, indent=4):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

def load_jsonl(file_path):
    data_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data_list.append(json.loads(line))
    return data_list

def save_jsonl(data_list, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data_list:
            line = json.dumps(item, ensure_ascii=False)
            f.write(line + "\n")

def save_to_jsonl(data, file_path):
    with open(file_path, "a", encoding="utf-8") as f:
        line = json.dumps(data, ensure_ascii=False)
        f.write(line + "\n")