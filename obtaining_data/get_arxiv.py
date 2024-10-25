import arxiv
import pandas as pd
from tqdm import tqdm

categories = ['cs.AI (Artificial Intelligence)',
 'cs.AR (Hardware Architecture)',
 'cs.CC (Computational Complexity)',
 'cs.CE (Computational Engineering, Finance, and Science)',
 'cs.CG (Computational Geometry)',
 'cs.CL (Computation and Language)',
 'cs.CR (Cryptography and Security)',
 'cs.CV (Computer Vision and Pattern Recognition)',
 'cs.CY (Computers and Society)',
 'cs.DB (Databases)',
 'cs.DC (Distributed, Parallel, and Cluster Computing)',
 'cs.DL (Digital Libraries)',
 'cs.DM (Discrete Mathematics)',
 'cs.DS (Data Structures and Algorithms)',
 'cs.ET (Emerging Technologies)',
 'cs.FL (Formal Languages and Automata Theory)',
#  'cs.GL (General Literature)',
 'cs.GR (Graphics)',
 'cs.GT (Computer Science and Game Theory)',
 'cs.HC (Human-Computer Interaction)',
 'cs.IR (Information Retrieval)',
 'cs.IT (Information Theory)',
 'cs.LG (Machine Learning)',
 'cs.LO (Logic in Computer Science)',
 'cs.MA (Multiagent Systems)',
 'cs.MM (Multimedia)',
 'cs.MS (Mathematical Software)',
 'cs.NA (Numerical Analysis)',
 'cs.NE (Neural and Evolutionary Computing)',
 'cs.NI (Networking and Internet Architecture)',
 'cs.OH (Other Computer Science)',
 'cs.OS (Operating Systems)',
 'cs.PF (Performance)',
 'cs.PL (Programming Languages)',
 'cs.RO (Robotics)',
 'cs.SC (Symbolic Computation)',
 'cs.SD (Sound)',
 'cs.SE (Software Engineering)',
 'cs.SI (Social and Information Networks)',
 'cs.SY (Systems and Control)']

num_articles_per_category = 100  # Change this number as needed


def get_articles_by_category(client, category, num_articles):
    """
    Fetch articles from arXiv for a specific category and starting from a specific date.
    
    Parameters:
    - client (arxiv.Client): The arXiv client instance.
    - category (str): The category to fetch articles from.
    - start_date (str): The starting date in YYYY-MM-DD format.
    - num_articles (int): The number of articles to fetch.
    
    Returns:
    - List of arXiv articles metadata.
    """
    query = f"cat:{category.split(' ')[0]} AND submittedDate:[2024 TO 2026]"
    print(query)
    search = arxiv.Search(
        query=query,
        max_results=num_articles,
        sort_by=arxiv.SortCriterion.Relevance
    )
    articles = [result for result in client.results(search)]
    return articles

def sample_articles(categories, num_articles_per_category):
    """
    Sample articles from multiple categories.

    Parameters:
    - categories (list): List of categories to sample articles from.
    - start_date (str): The starting date in YYYY-MM-DD format.
    - num_articles_per_category (int): Number of articles to sample per category.

    Returns:
    - Dict with category as key and list of articles as value.
    """
    client = arxiv.Client()
    dataset = {}
    for category in tqdm(categories):
        articles = get_articles_by_category(client, category, num_articles_per_category)
        dataset[category] = articles
    return dataset


dataset = sample_articles(categories, num_articles_per_category)

df = []
# Print out some details of the fetched articles for verification
for category, articles in dataset.items():
    for article in articles:
        print(article.published)
        df.append([article.title, article.published, article.authors, article.summary, category])
df = pd.DataFrame(df, columns=["Title", "Published", "Authors", "Summary", "Category"])
df.to_csv("data_in/arxiv_articles.csv", index=False)