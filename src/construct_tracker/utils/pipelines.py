
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier # TODO: add
from xgboost import XGBClassifier

def get_pipelines(feature_vector, model_name = 'Ridge', tfidf_vectorizer = None,random_state = 123):
	

	model = globals()[model_name]()
	model.set_params(random_state=random_state)
	
	if feature_vector =='tfidf':
		if tfidf_vectorizer == True:
			from sklearn.feature_extraction.text import TfidfVectorizer
		
		
			vectorizer = TfidfVectorizer(
					min_df=3, ngram_range=(1,2), 
					stop_words=None, #'english',# these include 'just': stopwords.words('english')+["'d", "'ll", "'re", "'s", "'ve", 'could', 'doe', 'ha', 'might', 'must', "n't", 'need', 'sha', 'wa', 'wo', 'would'], strip_accents='unicode',
					sublinear_tf=True,
					# tokenizer=nltk_lemmatize,
					token_pattern=r"(?u)\b\w\w+\b|!|\?|\"|\'",
						use_idf=True,
					)
			# alternative
			# from nltk import word_tokenize
			# from nltk.stem import WordNetLemmatizer
			# lemmatizer = WordNetLemmatizer()
			# def nltk_lemmatize(text):
			# 	return [lemmatizer.lemmatize(w) for w in word_tokenize(text)]
			# tfidf_vectorizer = TfidfVectorizer(tokenizer=nltk_lemmatize, stop_words='english')	
		pipeline = Pipeline([
			 ('vectorizer', vectorizer),
			 ('model', model), 
			])
	else:
		pipeline = Pipeline([
			('imputer', SimpleImputer(strategy='median')),
			('standardizer', StandardScaler()),
			 ('model', model), 
			])
	return pipeline




