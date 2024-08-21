
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.impute import SimpleImputer
import warnings



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






def get_params(feature_vector,model_name = 'Ridge', toy=False, ridge_alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000], ridge_alphas_toy = [0.1, 10]):
	if model_name in ['LogisticRegression']:
		if feature_vector == 'tfidf':
			if toy:
				warnings.warn('WARNING, running toy version')
				param_grid = {
				   'vectorizer__max_features': [256, 512],
				}
			else:
				param_grid = {
					'vectorizer__max_features': [512,2048,None],
					'model__C': ridge_alphas,
				}
	
		else:
			if toy:
				warnings.warn('WARNING, running toy version')
				param_grid = {
					'model__C': ridge_alphas_toy,
				}
			else:
				param_grid = {
					'model__C': ridge_alphas,
				}
	
	elif model_name in ['Ridge', 'Lasso']:
		if feature_vector == 'tfidf':
			if toy:
				warnings.warn('WARNING, running toy version')
				param_grid = {
				   'vectorizer__max_features': [256, 512],
				}
			else:
				param_grid = {
					'vectorizer__max_features': [512,2048,None],
					'model__alpha': ridge_alphas,
				}
	
		else:
			if toy:
				warnings.warn('WARNING, running toy version')
				param_grid = {
					'model__alpha': ridge_alphas_toy,
				}
			else:
				param_grid = {
					'model__alpha': ridge_alphas,
				}
	

	elif model_name in [ 'LGBMRegressor', 'LGBMClassifier']:
		if toy:
			warnings.warn('WARNING, running toy version')
			param_grid = {
			   # 'vectorizer__max_features': [256,2048],
				# 'model__colsample_bytree': [0.5, 1],
				'model__max_depth': [10,20], #-1 is the default and means No max depth
		
			}
		else:
			if feature_vector =='tfidf':
				param_grid = {
					'vectorizer__max_features': [256,2048,None],
					'model__num_leaves': [30,45,60],
					'model__colsample_bytree': [0.1, 0.5, 1],
					'model__max_depth': [0,5,15], #0 is the default and means No max depth
					'model__min_child_weight': [0.01, 0.001, 0.0001],
					'model__min_child_samples': [10, 20,40], #alias: min_data_in_leaf
				   'vectorizer__max_features': [256, 512],
					}
			
			param_grid = {
				'model__num_leaves': [30,45,60],
				'model__colsample_bytree': [0.1, 0.5, 1],
				'model__max_depth': [0,5,15], #0 is the default and means No max depth
				'model__min_child_weight': [0.01, 0.001, 0.0001],
				'model__min_child_samples': [10, 20,40], #alias: min_data_in_leaf
		
			}

	
	elif model_name in [ 'XGBRegressor', 'XGBClassifier']:
		if toy:
			warnings.warn('WARNING, running toy version')
			param_grid = {
				'model__max_depth': [10,20], #-1 is the default and means No max depth
		
			}
		else:
			if feature_vector =='tfidf':
				param_grid = {
					'vectorizer__max_features': [256,2048,None],
					'model__colsample_bytree': [0.1, 0.5, 1],
					'model__max_depth': [5,15, None], #None is the default and means No max depth
					'model__min_child_weight': [0.01, 0.001, 0.0001],
				
				   
					}
			
			param_grid = {
				'model__colsample_bytree': [0.1, 0.5, 1],
				'model__max_depth': [5,15, None], #None is the default and means No max depth
				'model__min_child_weight': [0.01, 0.001, 0.0001],
		
			}

	return param_grid





		
def get_combinations(parameters):
	"""
	parameters =   {'model__colsample_bytree': [1, 0.5, 0.1],
				'model__max_depth': [-1,10,20], #-1 is the default and means No max depth
				'model__min_child_weight': [0.01, 0.001, 0.0001],
				'model__min_child_samples': [10, 20,40], #alias: min_data_in_leaf
			   }
		
	from itertools import product
	combinations = list(product(*parameters.values()))
	"""
	
	parameter_set_combinations = []
	for combination in combinations:
		parameter_set_i = {}
		
		for i, k in enumerate(parameters.keys()):
			parameter_set_i[k] = combination[i]
		parameter_set_combinations.append(parameter_set_i)
	return parameter_set_combinations



