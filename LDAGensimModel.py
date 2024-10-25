from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases, Phraser
from nltk.stem import WordNetLemmatizer
from TopicModelingInterface import TopicModelingInterface

class LDAGensimModel(TopicModelingInterface):
    def __init__(self, config):
        super().__init__(config)
        self.dictionary = None
        self.lda_model = None
        self.bigram = None
        self.trigram = None
        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, texts):
        # Tokenize
        texts = [simple_preprocess(text) for text in texts]
        
        # Remove stopwords
        texts = [[word for word in doc if word not in STOPWORDS] for doc in texts]
        # Lemmatize
        texts = [[self.lemmatizer.lemmatize(word) for word in doc] for doc in texts]
        
        return texts

    def fit_transform(self, documents):
        # Preprocess the documents
        processed_docs = self.preprocess(documents)

        # Create Dictionary
        self.dictionary = corpora.Dictionary(processed_docs)
        self.dictionary.filter_extremes(keep_n=1_000)

        # Create Corpus
        corpus = [self.dictionary.doc2bow(doc) for doc in processed_docs]

        # Train LDA model
        self.lda_model = LdaModel(
            corpus=corpus,
            id2word=self.dictionary,
            num_topics=self.config["N_TOPICS"],
            random_state=self.config["SEED"],
            alpha='auto',
            eta='auto',
        )

        # Assign topics to documents
        topics = []
        topic_names = []
        for doc in corpus:
            topic_dist = self.lda_model.get_document_topics(doc)
            dominant_topic = max(topic_dist, key=lambda x: x[1])[0]
            topics.append(dominant_topic)
            
            # Get topic name (top 4 words)
            topic_words = self.lda_model.show_topic(dominant_topic, topn=4)
            topic_name = " ".join([word for word, _ in topic_words])
            topic_names.append(topic_name)

        num_topics = self.lda_model.num_topics

        return topics, topic_names, num_topics

    def get_topic_words(self, num_words=10):
        return {i: [word for word, _ in self.lda_model.show_topic(i, topn=num_words)] for i in range(self.lda_model.num_topics)}

    def get_coherence_score(self):
        from gensim.models.coherencemodel import CoherenceModel
        coherence_model = CoherenceModel(model=self.lda_model, texts=self.processed_docs, dictionary=self.dictionary, coherence='c_v')
        return coherence_model.get_coherence()