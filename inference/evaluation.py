from argparse import ArgumentParser
from tqdm import tqdm
import re

from inference.react_agent import MultiTurnReactAgent
from inference.Qwen2_5 import Qwen2_5
from inference.inference_util import *


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


def evaluate(agent, benchmark, llm_judge, args):
    cnt_total = len(benchmark)
    cnt_correct = 0
    result_list = []

    for data in tqdm(benchmark):
        result = agent._run(data)
        grade_result = grade_sample(llm_judge, result["question"], result["answer"], result["prediction"])
        if grade_result == "correct: yes":
            cnt_correct += 1
            result["correct"] = True
        else:
            result["correct"] = False
        result_list.append(result)

    acc = cnt_correct / cnt_total
    print(f"cnt_total: {cnt_total}")
    print(f"cnt_correct: {cnt_correct}")
    print(f"Pass@1: {acc}")
    save_jsonl(result_list, args.result_jsonl_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-8B-Thinking")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--base_url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--benchmark_jsonl_path", type=str, required=True)
    parser.add_argument("--result_jsonl_path", type=str, required=True)

    args = parser.parse_args()

    agent = MultiTurnReactAgent(
        model=args.model,
        api_key=args.api_key,
        base_url = args.base_url
    )
    benchmark = load_jsonl(args.benchmark_jsonl_path)
    llm_judge = Qwen2_5()
    evaluate(agent, benchmark, llm_judge, args)