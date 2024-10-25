import pandas as pd

class NYTDataset:
    def __init__(self):
        self.df = pd.read_csv('data_in/ny_times_articles.csv')
        self.data = self.df['abstract'].tolist()
        idx_mapping = {x: i for i, x in enumerate(self.df['keyword'].unique())}
        self.target = [idx_mapping[x] for x in self.df['keyword']]
        self.target_names = {i: x for i, x in enumerate(self.df['keyword'].unique())}

class ArXivDataset:
    def __init__(self):
        self.df = pd.read_csv('data_in/arxiv_articles.csv')
        self.data = self.df['Summary'].tolist()
        idx_mapping = {x: i for i, x in enumerate(self.df['Category'].unique())}
        self.target = [idx_mapping[x] for x in self.df['Category']]
        self.target_names = {i: x for i, x in enumerate(self.df['Category'].unique())}
        
class PubmedDataset:
    def __init__(self):
        self.df = pd.read_csv('data_in/pubmed_articles.csv')
        self.data = self.df['abstract'].tolist()
        idx_mapping = {x: i for i, x in enumerate(self.df['mesh_subheading'].unique())}
        self.target = [idx_mapping[x] for x in self.df['mesh_subheading']]
        self.target_names = {i: x for i, x in enumerate(self.df['mesh_subheading'].unique())}


def get_nyt():
    return NYTDataset()

def get_arxiv():
    return ArXivDataset()

def get_pubmed():
    return PubmedDataset()



if __name__ == '__main__':
    arxiv = get_arxiv()
    nytimes = get_nyt()
    print(nytimes.target_names)