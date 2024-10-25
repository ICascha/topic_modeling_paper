from TopicModelingInterface import TopicModelingInterface
import tiktoken
from genai_functions import (
    complete_openai_request,
    chunk_documents,
    topic_creation_prompt,
    topic_elimination_prompt,
    topic_classification_prompt,
    complete_openai_request_parralel,
)
from itertools import chain
import random
import json


class GenAIMethod(TopicModelingInterface):
    def __init__(self, config):
        super().__init__(config)
        self.model = config["MODEL"]

    def fit_transform(self, documents):
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        encoding_function = lambda x: enc.encode(x)
        decoding_function = lambda x: enc.decode(x)
        
        chunks = chunk_documents(
            documents,
            encoding_function,
            decoding_function,
            self.token_limit,
            max_documents=self.n_documents // 8,
        )

        prompts = [topic_creation_prompt(chunk) for chunk in chunks]
        results = complete_openai_request_parralel(
            prompts, model=self.model, timeout=30, batch_size=10
        )
        topic_list = list(
            chain(
                *[
                    result["topics"]
                    for result in results
                    if result and isinstance(result, dict) and "topics" in result
                ]
            )
        )
        topic_list = [x.lower() for x in topic_list]
        topic_list = list(set(topic_list))
        
        history = []
        # Initialize the history with the original topic list
        history.append({
            "step": 0,
            "topics": topic_list[:],
            "parents": {topic: None for topic in topic_list}
        })
        step = 1
                
        while len(topic_list) > self.n_topics:
            prompt = topic_elimination_prompt(topic_list)
            response = complete_openai_request(prompt, model=self.model)
            elimated_topics = [topic_list[i].lower() for i in response["topic_pair"]]
            old_topic_list = topic_list[:]
            new_topic = response["new_topic"].lower()
            try:
                if len(elimated_topics) != 2:
                    raise Exception("Invalid number of topics to eliminate")
                topic_list.remove(elimated_topics[0])
                topic_list.remove(elimated_topics[1])
                topic_list.append(new_topic)
                print(f"Eliminated {elimated_topics} and added {response['new_topic']}")
                print(f"Step {step}, steps to go: {len(topic_list) - self.n_topics}")
                
                # Create the parents dictionary for the current step
                current_parents = {topic: [topic] for topic in topic_list}
                current_parents[new_topic] = elimated_topics
                
                # Add the new state to history
                history.append({
                    "step": step,
                    "topics": topic_list[:],
                    "parents": current_parents
                })
                step += 1
            except Exception as e:
                topic_list = old_topic_list
                random.shuffle(topic_list)
                
        # Save the history to a JSON file
        with open('topic_history.json', 'w') as f:
            json.dump(history, f, indent=4)

        prompts = [
            topic_classification_prompt(document, topic_list) for document in documents
        ]
        results = complete_openai_request_parralel(
            prompts, model=self.model, timeout=30, batch_size=50
        )

        topic_assignments = [self.assign_topic(result) for result in results]
        topic_names = [topic_list[i] if i >= 0 else "ERROR_NO_TOPIC" for i in topic_assignments]
        return topic_assignments, topic_names, self.n_topics

    def assign_topic(self, result):
        if result is None:
            return -3
        if "topic" not in result:
            return -2
        if result["topic"] not in range(self.n_topics):
            return -1
        return result["topic"]