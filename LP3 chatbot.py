import pandas as pd
import nltk
import re
from nltk.stem import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import pos_tag
from sklearn.metrics import pairwise_distances


df=pd.read_excel('C:/Users/Personal/Desktop/Internship 1/Cloud Counselage/dialog_talk_agent.xlsx')

def text_normalization(text):
    text=str(text).lower()
    spl_char_text=re.sub(r'[^ a-z]','',text)
    tokens=nltk.word_tokenize(spl_char_text)
    lema=wordnet.WordNetLemmatizer()
    tags_list=pos_tag(tokens,tagset=None)
    lema_words=[]
    for token,pos_token in tags_list:
        if pos_token.startswith('V'):
            pos_val='v'
        elif pos_token.startswith('J'):
            pos_val='a'
        elif pos_token.startswith('R'):
            pos_val='r'
        else:
            pos_val='n'
        lema_token=lema.lemmatize(token,pos_val)
        lema_words.append(lema_token)
    return " ".join(lema_words)

df['lemmatized_text']=df['Context'].apply(text_normalization)

tfidf=TfidfVectorizer()
x_tfidf=tfidf.fit_transform(df['lemmatized_text']).toarray()
df_tfidf=pd.DataFrame(x_tfidf,columns=tfidf.get_feature_names())

def chat_tfidf(text):
    lemma=text_normalization(text)
    tf=tfidf.transform([lemma]).toarray()
    cos=1-pairwise_distances(df_tfidf,tf,metric='cosine')
    index_value=cos.argmax()
    return df['Text Response'].loc[index_value]
