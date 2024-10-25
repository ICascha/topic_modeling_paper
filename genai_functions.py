from itertools import chain
from openai import OpenAI

import json
import aiohttp
import asyncio
import os

from Auxiliary import delay_execution, delay_execution_async
from dotenv import load_dotenv

load_dotenv(".env")

API_KEY = os.getenv("OPENAI_KEY")
client = OpenAI(api_key=API_KEY)
TEMPERATURE = 0


def chunk_documents(
    documents, tokenizer_function, detokenizer_function, max_tokens, max_documents=10
):
    """This function splits a list of documents into chunks of size max_tokens."""
    chunks = [[]]
    current_num_tokens = 0
    for document in documents:
        tokens = len(tokenizer_function(document))
        if (current_num_tokens + tokens < max_tokens) and (
            len(chunks[-1]) < max_documents
        ):
            chunks[-1].append(document)
            current_num_tokens += tokens
        elif tokens >= max_tokens:
            truncated_document = detokenizer_function(
                tokenizer_function(document)[:max_tokens]
            )
            chunks.append([truncated_document])
            current_num_tokens = max_tokens
        else:
            chunks.append([document])
            current_num_tokens = tokens
    return chunks


def topic_creation_prompt(documents, type="news articles"):
    """This function takes a list of documents and returns a prompt that can be used to return a list of topics."""

    prompt = f"""Your task will be to distill a list of topics from the following {type}:\n\n"""
    prompt += " DOCUMENT: " + "\n DOCUMENT: ".join(documents) + "\n\n"
    prompt += (
        "Your response should be a JSON in the following format: {\"topics\": [\"topic1\", \"topic2\", \"topic3\"]}"
        + "\n"
    )
    prompt += "Topics should not be too specific, but also not too general. For example, 'food' is too general, but 'lemon cake' is too specific.\n"
    prompt += "A topic does not need to be present in multiple documents. But do not create more topics than there are documents, so if there are N documents, you should at most create N topics." + "\n"

    return prompt

def topic_creation_prompt_old(documents, type="news articles"):
    """This function takes a list of documents and returns a prompt that can be used to return a list of topics."""

    prompt = f"""Your task will be to distill a list of topics from the following {type}:\n\n"""
    prompt += "\n DOCUMENT: ".join(documents) + "\n\n"
    prompt += (
        "Your response should be a JSON in the following format: {\"topics\": [\"topic1\", \"topic2\", \"topic3\"]}"
        + "\n"
    )
    prompt += "Topics should not be too specific, but also not too general. For example, 'food' is too general, but 'lemon cake' is too specific."
    prompt += "A topic does not need to be present in multiple documents." + "\n"

    return prompt


def topic_combination_prompt(topic_list, n_topics):

    prompt = "Your task will be too distill a list of core topics from the following topics:\n\n"
    prompt += "\n TOPIC:".join(topic_list) + "\n\n"
    prompt += (
        "Your response should be a JSON in the following format: {\"topics\": [\"topic1\", \"topic2\", \"topic3\"]}"
        + "\n"
    )
    prompt += "Remove duplicate topics and merge topics that are too general. Merge topics together that are too specific. For example, 'food' might too general, but 'lemon cake' might too specific."
    prompt += f"In the end, try to arrive at a list of about {n_topics} topics."

    return prompt


def topic_combination_prompt_noprior(topic_list):

    prompt = "Your task will be too distill a list of core topics from the following topics:\n\n"
    prompt += "\n TOPIC:".join(topic_list) + "\n\n"
    prompt += (
        "Your response should be a JSON in the following format: {\"topics\": [\"topic1\", \"topic2\", \"topic3\"]}"
        + "\n"
    )
    prompt += "Remove duplicate topics and merge topics that are too general. Merge topics together that are too specific. For example, 'food' might too general, but 'lemon cake' might too specific. Arrive at a reasonable amount of core topics, whatever best suits the data."

    return prompt


def topic_classification_prompt(document, topics):
    topics = enumerate(topics)
    prompt = f"Your task will be to classify the following document into one of the following topics:\n\n"
    prompt += f"DOCUMENT: {document}\n\n"
    prompt += "\n".join([f"#{index}: {topic}" for index, topic in topics]) + "\n\n"
    prompt += (
        "Your response should be a JSON in the following format: {\"topic\": idx} with idx integer." + "\n"
    )
    prompt += "The index should be the index of the topic in the list of topics."
    return prompt


def topic_elimination_prompt_oldest(topics):
    topics = enumerate(topics)
    prompt = (
        f"Your task will be to merge a pair of topics out of the following topics:\n\n"
    )
    prompt += "\n".join([f"#{index}: {topic}" for index, topic in topics]) + "\n\n"
    prompt += "Your response should be a JSON in the following format: {\"topic_pair\": [idx1, idx2], \"new_topic\": \"new_topic\"}\n with idx1, idx2 integers."
    prompt += "The index should be the index of the topic in the list of topics.\n"
    prompt += "The new topic should be a combination of the two topics. Keep the name of the topic simple, try to generalize. So if you merge topic 'A' and 'B' together, do not name the topic something like 'A and B'. Rather, find the common more general denominator.\n"
    prompt += "In selecting the pair to merge, please merge the most similar, and most granular topics first."
    prompt += "Your topic set may also be 'poisoned' with a few topics that are too general. For example, a topic may be named 'A and B' when A and B do not have a strong relationship. If you see such a topic, please merge it with the most similar and appropriate topic to select one of the two."
    return prompt


def topic_elimination_prompt(topics):
    topics = enumerate(topics)
    prompt = (
        f"Your task will be to merge a pair of topics out of the following topics because the current topics are too granular:\n\n"
    )
    prompt += "\n".join([f"#{index}: {topic}" for index, topic in topics]) + "\n\n"
    prompt += "Your response should be a JSON in the following format: {\"topic_pair\": [idx1, idx2], \"new_topic\": \"new_topic\"}\n with idx1, idx2 integers."
    prompt += "The index should be the index of the topic in the list of topics.\n"
    prompt += "The new topic should be a generalization of the two topics. Keep the name of the topic simple, try to generalize. So if you merge topic 'A' and 'B' together, do not name the topic something like 'A and B'. Rather, find the common more general denominator.\n"
    prompt += "In selecting the pair to merge, please merge the most similar, and most granular topics first."
    prompt += "If you encounter a topic that is too general (e.g., 'A and B' without A and B having a strong relationship), merge it with the most appropriate and similar topic to create a more specific topic instead of generalizing."
    return prompt

def topic_elimination_prompt_weighted(topic_list, topic_weights):
    # zip the topics and weights together
    topics = enumerate(zip(topic_list, topic_weights))
    prompt = (
        f"Your task will be to merge a pair of topics out of the following topics because the current topics are too granular:\n\n"
    )
    prompt += "\n".join([f"#{index}: {topic}, weight: {weight}" for index, (topic, weight) in topics]) + "\n\n"
    prompt += "Your response should be a JSON in the following format: {\"topic_pair\": [idx1, idx2], \"new_topic\": \"new_topic\"}\n with idx1, idx2 integers."
    prompt += "The index should be the index of the topic in the list of topics.\n"
    prompt += "The new topic should be a generalization of the two topics. Keep the name of the topic simple, try to generalize. So if you merge topic 'A' and 'B' together, do not name the topic something like 'A and B'. Rather, find the common more general denominator.\n"
    prompt += "In selecting the pair to merge, please merge the most similar, and most granular topics first."
    prompt += "Your topic set may also be 'poisoned' with a few topics that are too general. For example, a topic may be named 'A and B' when A and B do not have a strong relationship. If you see such a topic, please merge it with the most similar and appropriate topic to select one of the two."
    prompt += "The process of merging topics is iterative. You have already merged some topics. Thus, each topic has a 'weight', this weight counts how many original topics have been merged into the topic. The weight is a measure of how general the topic is. The higher the weight, the more general the topic. When merging topics, please merge the topics with the lowest weight first, as these are two granular."
    return prompt


def topic_buildup_prompt(topics, curr_list, final_size):
    prompt = "In a previous task, you have identified a list of topics. This list is too large; it contains many topics that are near-duplicates or too granular. Your task is to create a smaller more general list of core topics.\n"
    prompt += "The following topics have been identified:\n\n"
    prompt += "\n".join(topics) + "\n\n"
    prompt += "We need to reach a list of core topics that has size " + str(final_size) + ".\n"
    prompt += "You are in the process of building the smaller list of core topics. The current list is:\n\n"
    prompt += "\n".join(curr_list) + "\n\n"
    prompt += "Currently, the list has size " + str(len(curr_list)) + ".\n"
    prompt += "So we still need to add " + str(final_size - len(curr_list)) + " topics.\n"
    prompt += "We will expand the list by one topic for now. Please provide a new topic that is not already in the list.\n"
    prompt += "Your response should be a JSON in the following format: {\"new_topic\": \"new_topic\"}\n"
    return prompt
    

@delay_execution(seconds=5, tries=2)
def complete_openai_request(prompt, model="gpt-4o", timeout=30, temperature=0):
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant designed to output JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        timeout=timeout,
        temperature=temperature,
        max_tokens=2_000,
    )

    json_string = response.choices[0].message.content
    json_dict = json.loads(json_string)
    return json_dict


@delay_execution_async(seconds=5, tries=30)
async def complete_openai_request_http(session, prompt, model, timeout):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    data = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant designed to output JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": TEMPERATURE,
    }
    timeout = aiohttp.ClientTimeout(
        total=timeout
    )  # Set the total timeout for the whole operation
    async with session.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=data,
        timeout=timeout,
    ) as response:
        if response.status != 200:
            print(await response.text())
            print(prompt[:200])
            response.raise_for_status()  # This will raise an exception for HTTP errors
        response_json = await response.json()
        response_json = response_json["choices"][0]["message"]["content"]
        return json.loads(response_json)
     
@delay_execution_async(seconds=5, tries=30)
async def complete_openai_request_http_logprobs(session, prompt, model, timeout):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    data = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant designed to output JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": TEMPERATURE,
        "logprobs": True,
        "top_logprobs": 20,
    }
    timeout = aiohttp.ClientTimeout(
        total=timeout
    )  # Set the total timeout for the whole operation
    async with session.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=data,
        timeout=timeout,
    ) as response:
        if response.status != 200:
            print(await response.text())
            print(prompt[:200])
            response.raise_for_status()  # This will raise an exception for HTTP errors
        response_json = await response.json()
        return response_json

def complete_openai_request_parralel(
    prompts, model="gpt-3.5-turbo", timeout=30, batch_size=100, logprobs=False
):

    async def parralel_openai_request(prompts, model, timeout, batch_size):
        async with aiohttp.ClientSession() as session:
            if logprobs:
                  tasks = [
                     complete_openai_request_http_logprobs(session, prompt, model, timeout)
                     for prompt in prompts
                  ]
            else:
                  tasks = [
                complete_openai_request_http(session, prompt, model, timeout)
                for prompt in prompts
            ]
            all_objects = []
            # gather preserves the order of the tasks
            for i in range(0, len(tasks), batch_size):
                responses = await asyncio.gather(
                    *tasks[i : i + batch_size], return_exceptions=True
                )
                for response in responses:
                    if isinstance(response, Exception):
                        # return None if there was an error
                        all_objects.append(None)
                    else:
                        all_objects.append(response)
                await asyncio.sleep(5)

            return all_objects

    responses = asyncio.run(
        parralel_openai_request(prompts, model, timeout, batch_size)
    )
    return responses