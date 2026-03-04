import argparse
import traceback
from tqdm import tqdm
import json

from Qwen2_5 import Qwen2_5
from data_synthesis_util import *
from offline_wikipedia import obtain_entity_information


def generate_question(seed, qwen2_5):
    print(f"Seed: {seed}")
    print()

    entity_information = obtain_entity_information(seed)
    link = entity_information["article_url"]

    print(f"Link: {link}")
    print()

    page_content = entity_information["text"]

    prompt = """
    generate a question and answer pair based on the page content and the provided entity.

    the question should be about the provided entity.
    question should be one sentence.
    the question should be specific so that there exist only one answer.
    the answer should be a word or a short phrase. which is easy to verify correctness.
    the answer should be concise and specific.
    return the question and answer pair in json format.
    the json format should be like:
    {
        "question": ...,
        "answer": ...
    }
    do not provide any other information.


    page content:
    [PAGE CONTENT]

    entity:
    [ENTITY]
    """

    prompt = prompt.replace("[PAGE CONTENT]", page_content)
    prompt = prompt.replace("[ENTITY]", seed)

    question_and_answer = qwen2_5.query(prompt)

    print(f"Question and answer: {question_and_answer}")
    print()

    question_and_answer = question_and_answer.replace("```json", "").replace("```", "").strip()
    question_and_answer_json = json.loads(question_and_answer)

    print(f"Question: {question_and_answer_json['question']}")
    print()
    print(f"Answer: {question_and_answer_json['answer']}")
    print()

    return question_and_answer_json

def question_transformation(question, qwen2_5, generated_data, iteration):
    prompt = """
    select a entity from the given question.

    only output the entity, no need for other explantion.
    the entity should be substring of the given question, do not answer the question.

    example:
    Ukraine, World War II, Ivy League, PageRank, Sverker Johansson, Virginia Woolf, Reykjavík, Borobudur

    do not select entity like this:
    German-born theoretical physicist, Polish-born, theory, Roman emperor from 98 to 117 AD, 

    text: [TEXT]
    """

    prompt = prompt.replace("[TEXT]", question)

    entity = qwen2_5.query(prompt)
    generated_data["iteration_" + str(iteration) + "_entity"] = entity

    print(f"Picked entity: {entity}")
    print()

    entity_information = obtain_entity_information(entity)
    link = entity_information["article_url"]

    print(f"Link: {link}")
    print()

    page_content = entity_information["text"]

    prompt = """
    parse information of the entity from the given text.

    the information should be rarely known.
    the information should contain more entities as possible.
    the information should be short and concise, in one sentence.
    only output the information, no need for other explantion.

    parse information like this:
    First-lap contact with Lewis Hamilton dropped Lando Norris to the back at the Spanish Grand Prix 2023.

    A member of the McLaren Young Driver Programme since 2017, Lando Norris joined McLaren in 2019 to partner Carlos Sainz Jr..


    do not parse information like this:
    Vladimir Putin has served as President of Russia since 2012.

    As observational evidence for a dynamic universe was lacking at the time, Einstein introduced a new term.

    Following the discovery of the recession of the galaxies by Edwin Hubble in 1929, Einstein abandoned his static model of the universe, and proposed two dynamic models of the cosmos, the Friedmann–Einstein universe of 1931  and the Einstein–de Sitter universe of 1932. In each of these models, Einstein discarded the cosmological constant, claiming that it was "in any case theoretically unsatisfactory".

    text:
    [TEXT]

    entity:
    [ENTITY]
    """

    prompt = prompt.replace("[TEXT]", page_content)
    prompt = prompt.replace("[ENTITY]", entity)

    information = qwen2_5.query(prompt)

    print(f"Parsed information: {information}")
    print()

    prompt = """
    Transform the question based on the entity and its information.

    The entity should be hidden from the question.
    Replace the entity with its information.
    only output the transformed question, no need for other explantion.

    Use the part of the information that is less known.

    Example:
    Full Information:
    Founded by Jimmy Wales and Larry Sanger in 2001, Wikipedia began as a complementary project for Nupedia, switching from its own license to the GNU Free Documentation License at the urging of Richard Stallman.

    Use information like this:
    switching from its own license to the GNU Free Documentation License at the urging of Richard Stallman.

    Do not use information like this:
    Founded by Jimmy Wales and Larry Sanger in 2001

    Use the part of the information that contains more entities.

    Example:
    Full Information:
    The Mojave phone booth, originally set up in 1948 for volcanic cinder miners, became an internet sensation in 1997 after being featured in a New York Times article, leading to its removal by Pacific Bell in 2000 due to environmental concerns.

    Use information like this:
    became an internet sensation in 1997 after being featured in a New York Times article, leading to its removal by Pacific Bell in 2000 due to environmental concerns.

    Do not use information like this:
    The Mojave phone booth, originally set up in 1948 for volcanic cinder miners

    Question:
    [QUESTION]

    Entity:
    [ENTITY]

    Information:
    [INFORMATION]
    """

    prompt = prompt.replace("[QUESTION]", question)
    prompt = prompt.replace("[ENTITY]", entity)
    prompt = prompt.replace("[INFORMATION]", information)

    transformed_question = qwen2_5.query(prompt)

    print(f"Transformed question: {transformed_question}")
    print()

    return transformed_question


def handle_image(question, qwen2_5, generated_data):
    prompt = """
    select a entity from the given question.

    only output the entity, no need for other explantion.
    the entity should be substring of the given question, do not answer the question.
    select the entity that is more critical to the correct answering of the question.

    example:
    Ukraine, World War II, Ivy League, PageRank, Sverker Johansson, Virginia Woolf, Reykjavík, Borobudur

    do not select entity like this:
    German-born theoretical physicist, Polish-born, theory, Roman emperor from 98 to 117 AD, 

    text: [TEXT]
    """

    prompt = prompt.replace("[TEXT]", question)

    image_entity = qwen2_5.query(prompt)
    generated_data["image_entity"] = image_entity

    print(f"Picked image entity: {image_entity}")
    print()

    entity_information = obtain_entity_information(image_entity)
    link = entity_information["article_url"]

    print(f"Link: {link}")

    image_url = entity_information["image_url"]
    
    print(f"Image URL: {image_url}")
    print()

    prompt = """
    Transform the question based on the entity.

    The entity should be hidden from the question.
    Replace the entity with phase like 'shown in the image'.
    The question should be rephrased to keep logically coherent and easy to understand.

    Question:
    [QUESTION]

    Entity:
    [ENTITY]
    """

    prompt = prompt.replace("[QUESTION]", question)
    prompt = prompt.replace("[ENTITY]", image_entity)

    transformed_question = qwen2_5.query(prompt)

    print(f"Transformed question: {transformed_question}")
    print()

    return transformed_question, image_url

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_path", type=str, required=True)
    parser.add_argument("--n_iteration", type=int, default=5)
    parser.add_argument("--result_path", type=str, required=True)
    args = parser.parse_args()

    wikidata_entities = load_jsonl(args.seed_path)

    seeds = [entity["title"] for entity in wikidata_entities]

    qwen2_5 = Qwen2_5()

    print("Begin data synthesis process")
    print(f"Number of seeds: {len(seeds)}")

    for seed in tqdm(seeds):
        print(f"Processing seed: {seed}")

        try:
            generated_data = {}
            generated_data["seed"] = seed

            question_and_answer_json = generate_question(seed, qwen2_5)
            question = question_and_answer_json["question"]
            answer = question_and_answer_json["answer"]
            generated_data["inital_question"] = question
            generated_data["answer"] = answer
            print('--------------------------------')

            print(f"Original question: {question}")
            print('--------------------------------')

            n_iteration = args.n_iteration

            for i in range(n_iteration):
                print(f"Iteration {i+1}")
                print()
                transformed_question = question_transformation(question, qwen2_5, generated_data, i)
                question = transformed_question
                print('--------------------------------')

            transformed_question, image_url = handle_image(transformed_question, qwen2_5, generated_data)
            if image_url is None:
                continue
            generated_data["question"] = transformed_question
            generated_data["image_path"] = image_url
            save_to_jsonl(generated_data, args.result_path)
        except Exception as e:
            print(f"Catch exception when handle seed: {seed}")
            print(traceback.format_exc())