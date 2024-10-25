import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.cluster import v_measure_score, homogeneity_score, completeness_score, adjusted_mutual_info_score
import tiktoken
from collections import defaultdict
from Datasets import get_nyt, get_arxiv, get_pubmed


class TopicModelingInterface:
    def __init__(self, config):
        self.config = config
        self.seed = config["SEED"]
        self.n_runs = config["N_runs"]
        self.n_documents = config["N_documents"]
        self.n_topics = config["N_TOPICS"]
        self.token_limit = config["TOKEN_LIMIT"]
        self.dataset = config["DATASET"]
        self.random_state = config["SEED"]

    def preprocess_documents(self, documents):
        raise NotImplementedError

    def fit_transform(self, documents):
        raise NotImplementedError

    def get_topic_info(self):
        raise NotImplementedError

    def run(self):
        random.seed(self.seed)
        score_df = []
        topic_name_df = []
        for counter in tqdm(range(self.n_runs)):
            enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

            if self.dataset == "NYT":
                newsgroups_train = get_nyt()
            elif self.dataset == "ARXIV":
                newsgroups_train = get_arxiv()
            elif self.dataset == "PUBMED":
                newsgroups_train = get_pubmed()   
            else:
                newsgroups_train = fetch_20newsgroups(
                    subset="train", remove=("headers", "footers", "quotes")
                )

            filtered_data_indices = [
                i
                for i, document in enumerate(newsgroups_train.data)
                if len(document) > 0 and len(enc.encode(document)) < self.token_limit
            ]
            filtered_data = [newsgroups_train.data[i] for i in filtered_data_indices]
            filtered_labels = [
                newsgroups_train.target[i] for i in filtered_data_indices
            ]
            # filter randomly self.n_documents indices
            indices = random.sample(
                range(len(filtered_data)), min(self.n_documents, len(filtered_data))
            )
            documents = [filtered_data[i] for i in indices]
            labels = [filtered_labels[i] for i in indices]
            
            ground_truth_names = [newsgroups_train.target_names[label] for label in labels]
            topics, topic_names, num_topics = self.fit_transform(documents)
            score = v_measure_score(labels, topics)
            score_df.append(("V_measure", score, num_topics))
            score_df.append(("completeness", completeness_score(labels, topics), num_topics))
            score_df.append(("homogeneity", homogeneity_score(labels, topics), num_topics))
            score_df.append(("adjusted_mutual_info", adjusted_mutual_info_score(labels, topics), num_topics))
            topic_name_df.append(pd.DataFrame({"topic_name": topic_names, "ground_truth": ground_truth_names, "run": counter}))

            score_df_out = pd.DataFrame(
                score_df, columns=["metric_name", "score", "num_topics"]
            )
            score_df_out.to_csv(
                f"data_out/coherence_scores_{self.__class__.__name__}_{self.n_documents}_{self.dataset}.csv"
            )
            pd.concat(topic_name_df).to_csv(
                f"data_out/topic_names_{self.__class__.__name__}_{self.n_documents}_{self.dataset}.csv"
            )

    def sample_equal_per_class(self, data, labels, n_documents, random_state=None):
        if random_state is not None:
            random.seed(random_state)

        data_by_class = defaultdict(list)
        for datum, label in zip(data, labels):
            data_by_class[label].append(datum)

        num_classes = len(data_by_class)
        samples_per_class = n_documents // num_classes

        sampled_data = []
        sampled_labels = []

        for label, data_list in data_by_class.items():
            if len(data_list) < samples_per_class:
                raise ValueError(
                    f"Not enough instances in class {label} to sample {samples_per_class} instances"
                )
            sampled_indices = random.sample(range(len(data_list)), samples_per_class)
            sampled_data.extend(data_list[i] for i in sampled_indices)
            sampled_labels.extend(label for _ in sampled_indices)

        combined = list(zip(sampled_data, sampled_labels))
        random.shuffle(combined)
        sampled_data, sampled_labels = zip(*combined)

        return list(sampled_data), list(sampled_labels)
