import warnings
from skopt import BayesSearchCV # had to replace np.int for in in transformers.py

ridge_alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
ridge_alphas_toy = [0.1, 10]
def get_params(feature_vector,model_name = 'Ridge', toy=False):
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




def hyparameter_tuning(pipeline, parameters,X_train,y_train,  scoring, method = 'bayesian', cv=5, return_train_score=False,
					n_iter=32, random_state=123):

	if method == 'bayesian':
		pipeline = BayesSearchCV(pipeline, parameters, cv=cv, scoring=scoring, return_train_score=return_train_score,
		n_iter=n_iter, random_state=random_state)    
				
		pipeline.fit(X_train,y_train)
		best_params = pipeline.best_params_
		best_model = pipeline.best_estimator_

		# y_pred = best_model.predict(X_test)
		
		# Content validity
	return pipeline, best_params