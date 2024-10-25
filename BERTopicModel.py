import os
import openai
from bertopic.backend import OpenAIBackend
from bertopic import BERTopic
from TopicModelingInterface import TopicModelingInterface
from dotenv import load_dotenv

load_dotenv(".env")

class BERTopicModel(TopicModelingInterface):
    def __init__(self, config):
        super().__init__(config)
        API_KEY = os.getenv("OPENAI_KEY")
        client = openai.OpenAI(api_key=API_KEY)
        self.embedding_model = OpenAIBackend(client, "text-embedding-3-large")

    def fit_transform(self, documents):
        model = BERTopic(
            embedding_model=self.embedding_model,
            nr_topics=self.n_topics,
            min_topic_size=2,
        )
        topics, _ = model.fit_transform(documents)
        # obtain topic names
        topic_name_mapping = model.get_topic_info()['Name']
        if min(topics) < 0:
            topics = [topic + 1 for topic in topics]
        topic_names = [topic_name_mapping.loc[topic] for topic in topics]
        return topics, topic_names, len(topic_name_mapping)