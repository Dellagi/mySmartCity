# -*- coding: utf-8 -*-

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('sentiwordnet')


from wordcloud import WordCloud
from nltk import ngrams
from nltk.corpus import stopwords, sentiwordnet as swn
from nltk.stem import WordNetLemmatizer

from sklearn.decomposition import LatentDirichletAllocation
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from transformers import TFCamembertForSequenceClassification

import re, sys, collections, json, string, nltk, requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import tensorflow as tf
from transformers import CamembertTokenizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import sqlite3 as sql

import os
from bs4 import BeautifulSoup as bs



stop_words_arr = ['a', 'abord', 'absolument', 'afin', 'ah', 'ai', 'aie', 'aient', 'aies', 'ailleurs', 'ainsi', 'ait', 'allaient', 'allo', 'allons', 'allô', 'alors', 'anterieur', 'anterieure', 'anterieures', 'apres', 'après', 'as', 'assez', 'attendu', 'au', 'aucun', 'aucune', 'aucuns', 'aujourd', "aujourd'hui", 'aupres', 'auquel', 'aura', 'aurai', 'auraient', 'aurais', 'aurait', 'auras', 'aurez', 'auriez', 'aurions', 'aurons', 'auront', 'aussi', 'autant', 'autre', 'autrefois', 'autrement', 'autres', 'autrui', 'aux', 'auxquelles', 'auxquels', 'avaient', 'avais', 'avait', 'avant', 'avec', 'avez', 'aviez', 'avions', 'avoir', 'avons', 'ayant', 'ayez', 'ayons', 'b', 'bah', 'bas', 'basee', 'bat', 'beau', 'beaucoup', 'bien', 'bigre', 'bon', 'boum', 'bravo', 'brrr', 'c', 'car', 'ce', 'ceci', 'cela', 'celle', 'celle-ci', 'celle-là', 'celles', 'celles-ci', 'celles-là', 'celui', 'celui-ci', 'celui-là', 'celà', 'cent', 'cependant', 'certain', 'certaine', 'certaines', 'certains', 'certes', 'ces', 'cet', 'cette', 'ceux', 'ceux-ci', 'ceux-là', 'chacun', 'chacune', 'chaque', 'cher', 'chers', 'chez', 'chiche', 'chut', 'chère', 'chères', 'ci', 'cinq', 'cinquantaine', 'cinquante', 'cinquantième', 'cinquième', 'clac', 'clic', 'combien', 'comme', 'comment', 'comparable', 'comparables', 'compris', 'concernant', 'contre', 'couic', 'crac', 'd', 'da', 'dans', 'de', 'debout', 'dedans', 'dehors', 'deja', 'delà', 'depuis', 'dernier', 'derniere', 'derriere', 'derrière', 'des', 'desormais', 'desquelles', 'desquels', 'dessous', 'dessus', 'deux', 'deuxième', 'deuxièmement', 'devant', 'devers', 'devra', 'devrait', 'different', 'differentes', 'differents', 'différent', 'différente', 'différentes', 'différents', 'dire', 'directe', 'directement', 'dit', 'dite', 'dits', 'divers', 'diverse', 'diverses', 'dix', 'dix-huit', 'dix-neuf', 'dix-sept', 'dixième', 'doit', 'doivent', 'donc', 'dont', 'dos', 'douze', 'douzième', 'dring', 'droite', 'du', 'duquel', 'durant', 'dès', 'début', 'désormais', 'e', 'effet', 'egale', 'egalement', 'egales', 'eh', 'elle', 'elle-même', 'elles', 'elles-mêmes', 'en', 'encore', 'enfin', 'entre', 'envers', 'environ', 'es', 'essai', 'est', 'et', 'etant', 'etc', 'etre', 'eu', 'eue', 'eues', 'euh', 'eurent', 'eus', 'eusse', 'eussent', 'eusses', 'eussiez', 'eussions', 'eut', 'eux', 'eux-mêmes', 'exactement', 'excepté', 'extenso', 'exterieur', 'eûmes', 'eût', 'eûtes', 'f', 'fais', 'faisaient', 'faisant', 'fait', 'faites', 'façon', 'feront', 'fi', 'flac', 'floc', 'fois', 'font', 'force', 'furent', 'fus', 'fusse', 'fussent', 'fusses', 'fussiez', 'fussions', 'fut', 'fûmes', 'fût', 'fûtes', 'g', 'gens', 'h', 'ha', 'haut', 'hein', 'hem', 'hep', 'hi', 'ho', 'holà', 'hop', 'hormis', 'hors', 'hou', 'houp', 'hue', 'hui', 'huit', 'huitième', 'hum', 'hurrah', 'hé', 'hélas', 'i', 'ici', 'il', 'ils', 'importe', 'j', 'je', 'jusqu', 'jusque', 'juste', 'k', 'l', 'la', 'laisser', 'laquelle', 'las', 'le', 'lequel', 'les', 'lesquelles', 'lesquels', 'leur', 'leurs', 'longtemps', 'lors', 'lorsque', 'lui', 'lui-meme', 'lui-même', 'là', 'lès', 'm', 'ma', 'maint', 'maintenant', 'mais', 'malgre', 'malgré', 'maximale', 'me', 'meme', 'memes', 'merci', 'mes', 'mien', 'mienne', 'miennes', 'miens', 'mille', 'mince', 'mine', 'minimale', 'moi', 'moi-meme', 'moi-même', 'moindres', 'moins', 'mon', 'mot', 'moyennant', 'multiple', 'multiples', 'même', 'mêmes', 'n', 'na', 'naturel', 'naturelle', 'naturelles', 'ne', 'neanmoins', 'necessaire', 'necessairement', 'neuf', 'neuvième', 'ni', 'nombreuses', 'nombreux', 'nommés', 'non', 'nos', 'notamment', 'notre', 'nous', 'nous-mêmes', 'nouveau', 'nouveaux', 'nul', 'néanmoins', 'nôtre', 'nôtres', 'o', 'oh', 'ohé', 'ollé', 'olé', 'on', 'ont', 'onze', 'onzième', 'ore', 'ou', 'ouf', 'ouias', 'oust', 'ouste', 'outre', 'ouvert', 'ouverte', 'ouverts', 'o|', 'où', 'p', 'paf', 'pan', 'par', 'parce', 'parfois', 'parle', 'parlent', 'parler', 'parmi', 'parole', 'parseme', 'partant', 'particulier', 'particulière', 'particulièrement', 'pas', 'passé', 'pendant', 'pense', 'permet', 'personne', 'personnes', 'peu', 'peut', 'peuvent', 'peux', 'pff', 'pfft', 'pfut', 'pif', 'pire', 'pièce', 'plein', 'plouf', 'plupart', 'plus', 'plusieurs', 'plutôt', 'possessif', 'possessifs', 'possible', 'possibles', 'pouah', 'pour', 'pourquoi', 'pourrais', 'pourrait', 'pouvait', 'prealable', 'precisement', 'premier', 'première', 'premièrement', 'pres', 'probable', 'probante', 'procedant', 'proche', 'près', 'psitt', 'pu', 'puis', 'puisque', 'pur', 'pure', 'q', 'qu', 'quand', 'quant', 'quant-à-soi', 'quanta', 'quarante', 'quatorze', 'quatre', 'quatre-vingt', 'quatrième', 'quatrièmement', 'que', 'quel', 'quelconque', 'quelle', 'quelles', "quelqu'un", 'quelque', 'quelques', 'quels', 'qui', 'quiconque', 'quinze', 'quoi', 'quoique', 'r', 'rare', 'rarement', 'rares', 'relative', 'relativement', 'remarquable', 'rend', 'rendre', 'restant', 'reste', 'restent', 'restrictif', 'retour', 'revoici', 'revoilà', 'rien', 's', 'sa', 'sacrebleu', 'sait', 'sans', 'sapristi', 'sauf', 'se', 'sein', 'seize', 'selon', 'semblable', 'semblaient', 'semble', 'semblent', 'sent', 'sept', 'septième', 'sera', 'serai', 'seraient', 'serais', 'serait', 'seras', 'serez', 'seriez', 'serions', 'serons', 'seront', 'ses', 'seul', 'seule', 'seulement', 'si', 'sien', 'sienne', 'siennes', 'siens', 'sinon', 'six', 'sixième', 'soi', 'soi-même', 'soient', 'sois', 'soit', 'soixante', 'sommes', 'son', 'sont', 'sous', 'souvent', 'soyez', 'soyons', 'specifique', 'specifiques', 'speculatif', 'stop', 'strictement', 'subtiles', 'suffisant', 'suffisante', 'suffit', 'suis', 'suit', 'suivant', 'suivante', 'suivantes', 'suivants', 'suivre', 'sujet', 'superpose', 'sur', 'surtout', 't', 'ta', 'tac', 'tandis', 'tant', 'tardive', 'te', 'tel', 'telle', 'tellement', 'telles', 'tels', 'tenant', 'tend', 'tenir', 'tente', 'tes', 'tic', 'tien', 'tienne', 'tiennes', 'tiens', 'toc', 'toi', 'toi-même', 'ton', 'touchant', 'toujours', 'tous', 'tout', 'toute', 'toutefois', 'toutes', 'treize', 'trente', 'tres', 'trois', 'troisième', 'troisièmement', 'trop', 'très', 'tsoin', 'tsouin', 'tu', 'té', 'u', 'un', 'une', 'unes', 'uniformement', 'unique', 'uniques', 'uns', 'v', 'va', 'vais', 'valeur', 'vas', 'vers', 'via', 'vif', 'vifs', 'vingt', 'vivat', 'vive', 'vives', 'vlan', 'voici', 'voie', 'voient', 'voilà', 'voire', 'vont', 'vos', 'votre', 'vous', 'vous-mêmes', 'vu', 'vé', 'vôtre', 'vôtres', 'w', 'x', 'y', 'z', 'zut', 'à', 'â', 'ça', 'ès', 'étaient', 'étais', 'était', 'jamais', 'étant', 'état', 'étiez', 'étions', 'été', 'étée', 'étées', 'étés', 'êtes', 'être', 'ô']

websites_lst = {'edf.fr': "EDF", 'engie.fr': 'Engie', 'total.direct-energie.com': 'TotalDirect'}

reviews_df_arr = []

for wb_i, businessUnitDisplayName in websites_lst.items():
	try:
		print("[+] Scrapping {}".format(wb_i))
		count_page = 1
		while count_page:
			q_url = 'https://www.trustpilot.com/review/{}?languages=fr&page={}'.format(wb_i, count_page)
			print(q_url)
			r = requests.get(q_url)
			if count_page > 1 and '&page=' not in r.url:
				break
			print("[!] page: {}".format(count_page))
			html = bs(r.content.decode('utf8'), features="lxml")
			reviews = html.findAll('script', {"id": "__NEXT_DATA__"})[0]
			reviews_df = pd.DataFrame(json.loads(reviews.text)['props']['pageProps']['reviews'])
			reviews_df['publishedDate'] = reviews_df['dates'].apply(lambda x: x['publishedDate'])
			reviews_df['businessUnitDisplayName'] = reviews_df['dates'].apply(lambda x: businessUnitDisplayName)
			reviews_df_arr += [reviews_df]
			count_page += 1
	except:
		break


df = pd.concat(reviews_df_arr)
df = df[df['language']=='fr'][['title', 'text', 'rating', 'publishedDate', 'businessUnitDisplayName']]

os.system("rm -f reviews.db")
conn = sql.connect('reviews.db')
df.to_sql('reviews', conn)


conn = sql.connect('reviews.db')
df = pd.read_sql('SELECT * FROM reviews', conn)
df = df.drop('index', axis=1)


df['review_text'] = df['text'].str.lower()
df['review_text'] = df['review_text'].str.replace(r'(https|http)?:\/(\w|\.|\/|\?|\=|\&|\%)*\b','')
df['review_text'] = df['review_text'].str.replace(r'www\.\S+\.com','')
df['review_text'] = df['review_text'].str.replace('[{}]'.format(string.punctuation), ' ')
df['review_text'] = df['review_text'].str.replace('nul nul', '')
df['review_text'] = df['review_text'].str.replace('il faut', '')
df['review_text'] = df['review_text'].str.replace('qu il', '')
df['review_text'] = df['review_text'].str.replace(r'[^\x00-\x7F]+', '')
df['review_text'] = df['review_text'].str.replace(r'<.*?>', '')
df['review_text'] = df['review_text'].str.replace(r' +', ' ')
df['review_text'] = df['review_text'].str.replace('\n', ' ')
df['reviews_clean'] = df['review_text'].str.lower().str.replace("'", '').str.replace('[^\w\s]', ' ').str.replace(" \d+", " ").str.replace(' +', ' ').str.strip()

df['tokenise'] = df.apply(lambda row: nltk.word_tokenize(row[6]), axis=1)

stop_words = stopwords.words('french')
stop_words.extend(stop_words_arr)

df['clean_stops'] = df['tokenise'].apply(lambda x: [item for item in x if item not in stop_words])

wordnet_lemmatizer = WordNetLemmatizer()

df['lemmatize'] = df['clean_stops'].apply(lambda x: [wordnet_lemmatizer.lemmatize(y) for y in x]) 


vector_obj = CountVectorizer(analyzer = 'word', ngram_range = (2, 2))
vec_arr = []

for index, row in df.iterrows():
	vec_arr.append(", ".join(row[9]))


fitted_vec_mx = vector_obj.fit_transform(vec_arr)

lda_model = LatentDirichletAllocation(n_components = 10, random_state = 321, evaluate_every = -1, n_jobs = -1)
lda_output = lda_model.fit_transform(fitted_vec_mx)


criteria_strs = ["criteria" + str(i) for i in range(1, lda_model.n_components + 1)]
df_criteria_doc = pd.DataFrame(np.round(lda_output, 2), columns = criteria_strs)

df_criteria_doc['main_subject'] = (np.argmax(df_criteria_doc.values, axis=1)+1)
df = pd.merge(df, df_criteria_doc, left_index = True, right_index = True, how = 'outer')

df_criteria_doc = pd.DataFrame(np.round(lda_output, 2), columns=criteria_strs, index=['Doc' + str(i) for i in range(df.shape[0])])


df_criteria_doc['main_subject'] = np.argmax(df_criteria_doc.values, axis=1)
df_criteria_words = pd.DataFrame(lda_model.components_)

df_criteria_words.columns = vector_obj.get_feature_names()
df_criteria_words.index = criteria_strs

df_criteria_no = pd.DataFrame(df_criteria_words.idxmax())
df_scores = pd.DataFrame(df_criteria_words.max())

_ = pd.merge(df_criteria_no, df_scores, left_index=True, right_index=True)
_.columns = ['criteria', 'criteria_confidence_float']

extract_criteria = []

for i in _['criteria'].unique():
	group_crit_df = _.loc[_['criteria'] == i].reset_index()
	group_crit_df = group_crit_df.sort_values('criteria_confidence_float', ascending=False).head(1)
	group_crit_df['criteria'] = group_crit_df['criteria'].values[0][-1]
	extract_criteria.append([group_crit_df['criteria'].unique()[0], group_crit_df['index'].unique().tolist()[0]])


extract_criteria = pd.DataFrame(extract_criteria, columns=['main_subject', 'criteria_str'])
extract_criteria['main_subject']=extract_criteria['main_subject'].astype(int)

df = df[['title', 'text', 'reviews_clean', 'rating', 'publishedDate', 'businessUnitDisplayName', 'main_subject']]
df = df.merge(extract_criteria, on='main_subject')



def encode_reviews(tokenizer, reviews, max_length):
	token_ids = np.zeros(shape=(len(reviews), max_length), dtype=np.int32)
	for i, review in enumerate(reviews):
		encoded = tokenizer.encode(review, max_length=max_length, truncation=True)
		token_ids[i, 0:len(encoded)] = encoded
	attention_mask = (token_ids != 0).astype(np.int32)
	return {"input_ids": token_ids, "attention_mask": attention_mask}



MAX_SEQ_LEN = 150
model_name = "camembert-base"
tokenizer = CamembertTokenizer.from_pretrained(model_name)

model = TFCamembertForSequenceClassification.from_pretrained("jplu/tf-camembert-base")
model.load_weights('camembert_based.hdf5')


infer_reviews = np.array(df['reviews_clean'])
encoded_inf = encode_reviews(tokenizer, infer_reviews, MAX_SEQ_LEN)


start_time = time.time()
scores = model.predict(encoded_inf)
stop_time = time.time()


y_pred = np.argmax(scores[0], axis=1)

df['predictions'] = pd.Series(y_pred.tolist(), index=df.index)

final_df = df[['title', 'text', 'reviews_clean', 'rating', 'publishedDate', 'businessUnitDisplayName', 'main_subject']]

os.system("rm -f new_analysed_energy.db")
conn = sql.connect('new_analysed_energy.db')
final_df.to_sql('analysed_energy', conn)
