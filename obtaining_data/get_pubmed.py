import requests
from xml.etree import ElementTree as ET
import pandas as pd


def get_pmids(mesh_subheading, start_date, end_date, max_results=100):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    search_url = (
        f"{base_url}esearch.fcgi?db=pubmed&term={mesh_subheading}[sh]"
        f"&datetype=pdat&mindate={start_date}&maxdate={end_date}"
        f"&sort=relevance&retmax={max_results}&retmode=json"
    )
    response = requests.get(search_url)
    search_results = response.json()
    return search_results['esearchresult']['idlist']

def fetch_article_details(pmids):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    pmid_list = ','.join(pmids)
    fetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={pmid_list}&retmode=xml"
    print(fetch_url)
    response = requests.get(fetch_url)
    root = ET.fromstring(response.content)
    articles = []
    for article in root.findall('.//PubmedArticle'):
        details = {}
        
        # Extract title
        title = article.find('.//ArticleTitle')
        details['title'] = title.text if title is not None else None
        
        # Extract abstract
        abstract = article.find('.//AbstractText')
        details['abstract'] = abstract.text if abstract is not None else None
        
        # Extract authors
        authors = article.findall('.//Author')
        author_list = []
        for author in authors:
            last_name = author.find('LastName')
            fore_name = author.find('ForeName')
            if last_name is not None and fore_name is not None:
                author_list.append(f"{fore_name.text} {last_name.text}")
        details['authors'] = ", ".join(author_list) if author_list else None
        
        # Extract publication date
        pub_date = article.find('.//PubDate')
        if pub_date is not None:
            year = pub_date.find('Year')
            month = pub_date.find('Month')
            day = pub_date.find('Day')
            if year is not None and month is not None and day is not None:
                details['pub_date'] = f"{year.text}-{month.text}-{day.text}"
            elif year is not None and month is not None:
                details['pub_date'] = f"{year.text}-{month.text}"
            elif year is not None:
                details['pub_date'] = year.text
            else:
                details['pub_date'] = None
        else:
            details['pub_date'] = None
        
        # Only include articles with non-None abstract and publication date
        if details['abstract'] is not None and details['pub_date'] is not None:
            articles.append(details)
    return articles

def get_popular_abstracts(mesh_subheadings, start_date="2024/01/01", end_date="2025/12/31"):
    all_articles = []
    for subheading in mesh_subheadings:
        pmids = get_pmids(subheading, start_date, end_date)
        articles = fetch_article_details(pmids)
        for article in articles:
            article['mesh_subheading'] = subheading
        all_articles.extend(articles)
    return all_articles

# List of MeSH subheadings
mesh_subheadings = ['AB',
 'IR',
 'AD',
 'IS',
 'AE',
 'IP',
 'AG',
 'LJ',
 'AA',
 'ME',
 'AN',
 'MT',
 'AH',
 'MI',
 'AI',
 'MO',
 'BI',
 'NU',
 'BS',
 'OG',
 'BL',
 'PS',
 'CF',
 'PY',
 'CS',
 'PA',
 'CI',
 'PK',
 'CH',
 'PD',
 'CL',
 'PH',
 'CO',
 'PP',
 'CN',
 'PO',
 'CY',
 'PC',
 'DF',
 'PX',
 'DI',
 'RE',
 'DH',
 'RT',
 'DG',
 'RH',
 'DE',
 'SC',
 'DT',
 'ST',
 'EC',
 'SN',
 'ED',
 'SD',
 'EM',
 'SU',
 'EN',
 'TU',
 'EP',
 'TH',
 'ES',
 'TO',
 'EH',
 'TM',
 'ET',
 'TR',
 'GE',
 'TD',
 'GD',
 'UL',
 'HI',
 'UR',
 'IM',
 'VE',
 'IN',
 'VI']

# Retrieve popular abstracts for each MeSH subheading
popular_articles = get_popular_abstracts(mesh_subheadings)

# Create a DataFrame from the list of articles
df = pd.DataFrame(popular_articles)

# Save the DataFrame to a CSV file
csv_file_path = 'data_in/pubmed_articles.csv'
df.to_csv(csv_file_path, index=False)