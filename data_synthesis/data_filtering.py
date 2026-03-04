import re
import argparse
from tqdm import tqdm


from Qwen2_5 import Qwen2_5
from Qwen2_5_VL import Qwen2_5_VL
from data_synthesis_util import *


DIRECT_ANSWER_PROMPT = """
[QUESTION]

Please directly answer this question. Answer with a short phrase. Do not provide any other information.
""".strip()

GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available.
""".strip()

IMAGE_EVALUATION_PROMPT = """
I am currently building a training dataset for a multimodal deep-research agent. I want the images in the training samples to be complex so that they encourage the agent to perform image search to obtain more information. I want to filter out samples with images that are too simple.

Now, evaluate the given image.

Answer "yes" if the image is simple (e.g., it is the flag of the United States or contains only text that is easy to parse).
Answer "no" if the image is complex, information-rich, or hard to interpret.

Just provide the final answer, do not provide any other information.
""".strip()


def filter_answer_in_question(question: str, answer: str) -> bool:
    return answer in question


def grade_sample(grader_model, question: str, correct_answer: str, response: str) -> str:
    grader_prompt = GRADER_TEMPLATE.format(
            question=question,
            correct_answer=correct_answer,
            response=response,
    )
    grading_response = grader_model.query(grader_prompt)
    # print(grading_response)
    match = re.search(r"correct: (yes|no)", grading_response)
    return match.group(0) if match else "correct: no"


def filter_can_be_directly_answered(lvlm, grader_model, question: str, image: str, answer: str) -> bool:
    prompt = DIRECT_ANSWER_PROMPT.replace("[QUESTION]", question)
    response = lvlm.query(prompt, image)
    # print(response)

    grade_result = grade_sample(grader_model, question, answer, response)
    # print(grade_result)
    return grade_result == "correct: yes"


def filter_image_too_simple(lvlm, image: str) -> bool:
    response = lvlm.query(IMAGE_EVALUATION_PROMPT, image)
    # print(response)
    return response == "yes"


def filter_can_be_directly_answered_with_text(llm, grader_model, question: str, answer: str) -> bool:
    prompt = DIRECT_ANSWER_PROMPT.replace("[QUESTION]", question)
    response = llm.query(prompt)
    # print(response)

    grade_result = grade_sample(grader_model, question, answer, response)
    # print(grade_result)
    return grade_result == "correct: yes"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--labeled_data_path", type=str, required=True)
    parser.add_argument("--filtered_data_path", type=str, required=True)

    args = parser.parse_args()

    data_list = load_jsonl(args.data_path)

    lvlm = Qwen2_5_VL()
    llm = Qwen2_5()
    grader_model = Qwen2_5()

    for data in tqdm(data_list):
        question = data["question"]
        image_path = data["image_path"]
        answer = data["answer"]

        if filter_answer_in_question(question, answer):
            data["answer_in_question"] = True
        else:
            data["answer_in_question"] = False

        if filter_can_be_directly_answered(lvlm, grader_model, question, image_path, answer):
            data["can_be_directly_answered"] = True
        else:
            data["can_be_directly_answered"] = False

        if filter_image_too_simple(lvlm, image_path):
            data["image_too_simple"] = True
        else:
            data["image_too_simple"] = False

        if filter_can_be_directly_answered_with_text(llm, grader_model, question, answer):
            data["can_be_directly_answered_with_text"] = True
        else:
            data["can_be_directly_answered_with_text"] = False

    save_jsonl(data_list, args.labeled_data_path)

    filtered_data_list = []

    for data in data_list:
        if not data["answer_in_question"] and not data["can_be_directly_answered"] and not data["image_too_simple"] and not data["can_be_directly_answered_with_text"]:
            filtered_data_list.append(data)

    save_jsonl(filtered_data_list, args.filtered_data_path)