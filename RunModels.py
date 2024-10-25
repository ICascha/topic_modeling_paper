from BERTopicModel import BERTopicModel
from GenAIMethodOneShot import GenAIMethodOneShot
from GenAIMethod import GenAIMethod
from GenAIMethodOneShotNoPrior import GenAIMethodOneShotNoPrior


def run_models(config):
    models = [GenAIMethodOneShotNoPrior(config)]
    for model in models:
        model.run()

if __name__ == "__main__":
    config = {
        "SEED": 44,
        "N_runs": 5,
        "N_documents": 800,
        "N_TOPICS": 50,
        "TOKEN_LIMIT": 6_000,
        "DATASET": "NYT",
        "MODEL": "gpt-4o",
        "N_FEATURES": 1000,
    }
    run_models(config)
