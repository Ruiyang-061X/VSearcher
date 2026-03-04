import datetime
import json


def today_date():
    return datetime.date.today().strftime("%Y-%m-%d")

def clean_tool_call(text: str) -> str:
    # Find the first '{' – start of JSON
    start = text.find("{")
    if start == -1:
        print("No JSON object found inside <tool_call>")
        return text

    # Walk the string to find the matching closing '}' for that JSON object
    depth = 0
    end = None
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end is None:
        print("Could not find balanced JSON object")
        return text

    json_str = text[start:end + 1].strip()

    return json_str

def save_json(data, file_path, indent=4):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


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