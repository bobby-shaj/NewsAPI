import numpy
import spacy
from spacy.lang.en import English
from os import path
from newsapi import NewsApiClient
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

nlp_eng = spacy.load('en_core_web_lg')
newsapi = NewsApiClient (api_key='5ff95b643cca4f95bb4e2c69f113d8ab')

def get_keywords_eng(text):
    #print(text)
    pos_tag = ['PRON', 'VERB', 'NOUN', 'ADJ']
    result = []
    nlp = spacy.load('en_core_web_lg')
    tokens = nlp(text)
    for token in tokens:
        if (token.text in nlp.Defaults.stop_words or token.dep_ == "punct"):
            continue
        if (token.pos_ in pos_tag):
            result.append(token.text)
    return result



articles_from_api = newsapi.get_everything(q='coronavirus', language='en', from_param='2021-09-24',\
     to='2021-10-23', sort_by='relevancy')

data = []

for article in articles_from_api['articles']:
    title = article['title']
    description = article['description']
    content = article['content']
    date = article['publishedAt']
    data.append({'title':title, 'date':date, 'desc':description, 'content':content})

df = pd.DataFrame(data)
df = df.dropna()
df.to_csv('check.csv', index=False)

""" keywords = []
for content in df.loc[:, "content"]:
    keywords.append(get_keywords_eng(content)) """

content = df.loc[1, "content"]
keywords = []
keywords = get_keywords_eng(content)

text = (str(keywords))

wordcloud = WordCloud(max_font_size = 50, max_words=100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()