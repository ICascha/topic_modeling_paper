### Code for the experiments of Generative AI for Small Dataset Topic Modeling

Multiple topic modeling techniques are tested on datasets outside of GPT-4o's knowledge cutoff.
Each technique implements `TopicModelingInterface`. `GenAIMethod` is a computationally more expensive
strategy where a set of candidate topics is iteratively reduced. We disregarded this technique for the simpler
`GenAiMethodOneShot`, which is described as `TopicGen` in the paper.

To install, run `pip install -r requirements.txt`. The code is tested on Python 3.9.

To use TopicGen or BERTopicModel, an OPENAI API key is required. Please fill in `example.env` and rename it to `.env`.

To use your own data place it in `data_in` and modify `Datasets`. Datasets used in the study are provided in `data_in`.

To run, go to RunModels, please keep in mind the cost in terms of API calls can be high depending on docuemnt length and dataset size.
