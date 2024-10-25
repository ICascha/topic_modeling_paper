from TopicModelingInterface import TopicModelingInterface
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class NMFModel(TopicModelingInterface):
    def __init__(self, config):
        super().__init__(config)

    def fit_transform(self, documents):
        # Train the NMF model
        tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=self.config["N_FEATURES"], stop_words='english')
        tfidf = tfidf_vectorizer.fit_transform(documents)
        nmf = NMF(n_components=self.config["N_TOPICS"], random_state=self.config["SEED"]).fit(tfidf)
        # Assign topics to documents
        topic_distribution = nmf.transform(tfidf)
        topics = np.argmax(topic_distribution, axis=1)  # Get the index of the highest contribution topic for each document
        num_topics = len(set(topics))
                
        # Assign names to topics based on top words
        feature_names = tfidf_vectorizer.get_feature_names_out()
        top_words_per_topic = []
        for topic_idx, topic in enumerate(nmf.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-4:-1]]  # Get the top 4 words for this topic
            top_words_per_topic.append(" ".join(top_words))

        # Create a mapping from integer to topic name
        topic_name_mapping = {i: name for i, name in enumerate(top_words_per_topic)}

        # Create the list of topic names for each document
        topic_names = [topic_name_mapping[topic] for topic in topics]
        
        return topics, topic_names, num_topics