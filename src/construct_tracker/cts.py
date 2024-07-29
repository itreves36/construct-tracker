# """
# Author: Daniel Low
# License: Apache 2.0
# """


import concurrent.futures
import numpy as np
import pandas as pd
import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

def get_construct_embeddings(construct, construct_tokens_d, construct_embeddings_d):
	lexicon_tokens = construct_tokens_d.get(construct)
	if not isinstance(lexicon_tokens, list):
		raise AssertionError("to run method lexicon_*, the construct should be a list of prototypes")
	construct_embeddings = []
	for token in lexicon_tokens:
		token_embedding = construct_embeddings_d.get(token)
		if str(token_embedding) == "None":
			print("(you should FIX) Could not retrieve embedding for: ", token)
		else:
			construct_embeddings.append(token_embedding)
	return construct_embeddings

def process_document(doc_id, docs_embeddings_d, construct_embeddings_all, constructs, method, summary_stat, skip_nan, doc_id_col_name):
	doc_token_embeddings_i = docs_embeddings_d.get(doc_id)  # embeddings for a document
	doc_token_embeddings_i = np.array(doc_token_embeddings_i, dtype=float)

	if skip_nan and doc_token_embeddings_i is None:
		return None

	feature_vectors_doc = [doc_id]
	feature_vectors_doc_col_names = [doc_id_col_name]
	cosine_scores_docs_all = {}

	# compute cosine similarity between each construct and this document
	for construct in constructs:
		construct_embeddings = construct_embeddings_all.get(construct)  # embeddings for a construct
		
		# cosine similarity
		if method.startswith("word_"):
			assert len(construct_embeddings.shape) == 1
			if doc_token_embeddings_i.shape[0] == 0:  # happens when there is an empty str
				doc_token_embeddings_i = [np.zeros(construct_embeddings.shape[0])]
			cosine_scores_docs_i = cosine_similarity([construct_embeddings], doc_token_embeddings_i)
		else:
			
			cosine_scores_docs_i = cosine_similarity(construct_embeddings, doc_token_embeddings_i)
			
				# print("broke for the following doc ID (returning cosine_similarity 0):")
				# print(doc_id)
				# print()
				# cosine_scores_docs_i = [0]

		cosine_scores_docs_all[str(doc_id) + "_" + construct] = cosine_scores_docs_i

		# all summary stats for a single construct will be concatenated side by side
		summary_stats_doc_i = []
		summary_stats_name_doc_i = []

		for stat in summary_stat:
			function = getattr(np, stat)  # e.g. np.max
			doc_sim_stat = function(cosine_scores_docs_i)
			summary_stats_doc_i.append(doc_sim_stat)
			summary_stats_name_doc_i.append(construct + "_" + stat)

		feature_vectors_doc.extend(summary_stats_doc_i)
		feature_vectors_doc_col_names.extend(summary_stats_name_doc_i)

	feature_vectors_doc_df = pd.DataFrame([feature_vectors_doc], columns=feature_vectors_doc_col_names)
	return feature_vectors_doc_df, cosine_scores_docs_all

def measure(
	construct_tokens_d=None,
	construct_embeddings_d=None,
	docs_embeddings_d=None,
	method="lexicon",
	summary_stat=["max"],
	remove_stat_name_from_col_name=False,
	return_cosine_similarity=True,
	minmaxscaler=(0, 1),
	skip_nan=False,
	doc_id_col_name='doc_id'
):
	if isinstance(summary_stat, str):
		summary_stat = [summary_stat]
		
	construct_embeddings_all = {}
	constructs = construct_tokens_d.keys()
	
	for construct in constructs:
		if method.startswith("word"):
			construct_tokens = construct_tokens_d.get(construct)
			construct_embeddings = construct_embeddings_d.get(construct_tokens)
		elif method.startswith("lexicon") or method.startswith("avglexicon"):
			construct_embeddings = get_construct_embeddings(construct, construct_tokens_d, construct_embeddings_d)
			if method.startswith("avglexicon"):
				construct_embeddings = np.mean(construct_embeddings, axis=0)
		construct_embeddings = np.array(construct_embeddings, dtype=float)
		construct_embeddings_all[construct] = construct_embeddings

	feature_vectors_all = []
	cosine_scores_docs_all = {}

	print(f'computing similarity between {len(constructs)} constructs and {len(docs_embeddings_d.keys())} documents...')

	with concurrent.futures.ThreadPoolExecutor() as executor:
		futures = [executor.submit(process_document, doc_id, docs_embeddings_d, construct_embeddings_all, constructs, method, summary_stat, skip_nan, doc_id_col_name) for doc_id in docs_embeddings_d.keys()]
		for future in tqdm.tqdm(concurrent.futures.as_completed(futures)):
			doc_result, doc_cosine_scores = future.result()
			if doc_result is not None:
				feature_vectors_all.append(doc_result)
				cosine_scores_docs_all.update(doc_cosine_scores)

	feature_vectors_all = pd.concat(feature_vectors_all).reset_index(drop=True) 

	# Scale between 0 and 1 to follow output range of other classification models.
	if minmaxscaler is not None:
		scaler = MinMaxScaler()
		feature_cols = [col for col in feature_vectors_all.columns if any(string in col for string in summary_stat)]
		feature_vectors_all[feature_cols] = scaler.fit_transform(feature_vectors_all[feature_cols].values)

	if remove_stat_name_from_col_name:
		for stat in summary_stat:
			feature_vectors_all.columns = [n.replace(f'_{stat}', '') for n in feature_vectors_all.columns]

	if return_cosine_similarity:
		return feature_vectors_all, cosine_scores_docs_all
	else:
		return feature_vectors_all









# optimized but not faster
# ===================================
	


# import concurrent.futures
# import numpy as np
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.preprocessing import MinMaxScaler

# def get_construct_embeddings(construct, construct_tokens_d, construct_embeddings_d):
# 	lexicon_tokens = construct_tokens_d.get(construct)
# 	if not isinstance(lexicon_tokens, list):
# 		raise AssertionError("to run method lexicon_*, the construct should be a list of prototypes")
# 	construct_embeddings = []
# 	for token in lexicon_tokens:
# 		token_embedding = construct_embeddings_d.get(token)
# 		if str(token_embedding) == "None":
# 			print("(you should FIX) Could not retrieve embedding for: ", token)
# 		else:
# 			construct_embeddings.append(token_embedding)
# 	return construct_embeddings

# def process_documents(doc_ids, docs_embeddings_d, construct_embeddings_all, constructs, method, summary_stat, skip_nan, doc_id_col_name):
# 	results = []
# 	cosine_scores_docs_all = {}

# 	for doc_id in doc_ids:
# 		doc_token_embeddings_i = docs_embeddings_d.get(doc_id)  # embeddings for a document
# 		doc_token_embeddings_i = np.array(doc_token_embeddings_i, dtype=float)

# 		if skip_nan and doc_token_embeddings_i is None:
# 			continue

# 		feature_vectors_doc = [doc_id]
# 		feature_vectors_doc_col_names = [doc_id_col_name]

# 		for construct in constructs:
# 			construct_embeddings = construct_embeddings_all.get(construct)

# 			if doc_token_embeddings_i.size == 0:  # handles empty document embeddings
# 				cosine_scores_docs_i = np.zeros((len(construct_embeddings), 1))
# 			else:
# 				cosine_scores_docs_i = cosine_similarity(construct_embeddings, doc_token_embeddings_i)

# 			cosine_scores_docs_all[str(doc_id) + "_" + construct] = cosine_scores_docs_i

# 			summary_stats_doc_i = []
# 			summary_stats_name_doc_i = []
# 			for stat in summary_stat:
# 				function = getattr(np, stat)
# 				doc_sim_stat = function(cosine_scores_docs_i)
# 				summary_stats_doc_i.append(doc_sim_stat)
# 				summary_stats_name_doc_i.append(construct + "_" + stat)

# 			feature_vectors_doc.extend(summary_stats_doc_i)
# 			feature_vectors_doc_col_names.extend(summary_stats_name_doc_i)

# 		feature_vectors_doc_df = pd.DataFrame([feature_vectors_doc], columns=feature_vectors_doc_col_names)
# 		results.append(feature_vectors_doc_df)

# 	return results, cosine_scores_docs_all

# def measure(
# 	construct_tokens_d=None,
# 	construct_embeddings_d=None,
# 	docs_embeddings_d=None,
# 	method="lexicon",
# 	summary_stat=["max"],
# 	remove_stat_name_from_col_name=False,
# 	return_cosine_similarity=True,
# 	minmaxscaler=(0, 1),
# 	skip_nan=False,
# 	doc_id_col_name='doc_id',
# 	num_workers=None
# ):
# 	if isinstance(summary_stat, str):
# 		summary_stat = [summary_stat]

# 	construct_embeddings_all = {}
# 	constructs = construct_tokens_d.keys()
# 	for construct in constructs:
# 		construct_embeddings = get_construct_embeddings(construct, construct_tokens_d, construct_embeddings_d)
# 		if method == "avglexicon":
# 			construct_embeddings = np.mean(construct_embeddings, axis=0)
# 		construct_embeddings_all[construct] = np.array(construct_embeddings, dtype=float)

# 	feature_vectors_all = []
# 	cosine_scores_docs_all = {}
# 	doc_ids = list(docs_embeddings_d.keys())
# 	print(f'computing similarity between {len(constructs)} and {len(doc_ids)} documents...')

# 	with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
# 		# Create tasks for processing documents in batches
# 		tasks = []
# 		batch_size = len(doc_ids) // (num_workers or 1)
# 		for i in tqdm.tqdm(range(0, len(doc_ids), batch_size)):
# 			batch_doc_ids = doc_ids[i:i + batch_size]
# 			task = executor.submit(process_documents, batch_doc_ids, docs_embeddings_d, construct_embeddings_all, constructs, method, summary_stat, skip_nan, doc_id_col_name)
# 			tasks.append(task)

# 		for future in tqdm.tqdm(concurrent.futures.as_completed(tasks)):
# 			doc_results, doc_cosine_scores = future.result()
# 			feature_vectors_all.extend(doc_results)
# 			cosine_scores_docs_all.update(doc_cosine_scores)

# 	feature_vectors_all = pd.concat(feature_vectors_all).reset_index(drop=True)

# 	if minmaxscaler is not None:
# 		scaler = MinMaxScaler()
# 		feature_cols = [col for col in feature_vectors_all.columns if any(stat in col for stat in summary_stat)]
# 		feature_vectors_all[feature_cols] = scaler.fit_transform(feature_vectors_all[feature_cols].values)

# 	if remove_stat_name_from_col_name:
# 		for stat in summary_stat:
# 			feature_vectors_all.columns = [n.replace(f'_{stat}', '') for n in feature_vectors_all.columns]

# 	if return_cosine_similarity:
# 		return feature_vectors_all, cosine_scores_docs_all
# 	else:
# 		return feature_vectors_all

# ==============================



# import numpy as np
# import pandas as pd
# import tqdm
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.preprocessing import MinMaxScaler

# def get_construct_embeddings(construct,construct_tokens_d, construct_embeddings_d):
# 	lexicon_tokens = construct_tokens_d.get(construct)

# 	if not type(lexicon_tokens) == list:
# 		raise AssertionError("to run method lexicon_*, the construct should be a list of prototypes")

# 	construct_embeddings = []
# 	for token in lexicon_tokens:
# 		token_embedding = construct_embeddings_d.get(token)
# 		if str(token_embedding) == "None":
# 			print("(you should FIX) Could not retrieve embedding for: ", token)
# 		else:
# 			construct_embeddings.append(token_embedding)
# 	return construct_embeddings


# def measure(
# 	construct_tokens_d=None,
# 	construct_embeddings_d=None,
# 	docs_embeddings_d=None,
# 	method="lexicon",  # todo: change to token, tokens, weighted_tokens
# 	summary_stat=["max"],
# 	return_cosine_similarity=True,
# 	minmaxscaler=(0, 1),
# 	skip_nan=False,
# 	doc_id_col_name = 'doc_id'
# 	):
# 	"""
# 			A doc is composed of tokens. We compute the similarity between the construct and each token
# 	and taken some summary statistics
# 	Args:
# 			construct_tokens_d: 
# 				dictionary split by construct
# 				{construct1:[token1, token2, ...], construct2:[token1, token2, ...]}
# 			construct_embeddings_d: 
# 				single dictionary for all tokens, not split by construct
# 				{token1: embedding, token2: embedding, ...}
# 			docs_embeddings_d: 
# 				single dictionary where keys are document ids and values are the embeddings for each document's token/s
# 				{doc_id_001: [document_token1, document_token2, ...], doc_id_002: [document_token1, document_token2, ...]}
# 			method: {'word'', 'lexicon', 'avglexicon'}
# 			summary_stat: ['max', 'mean', 'median', 'sum']
# 	Returns: 
# 		feature_vectors_all: DataFrame of feature vectors
# 		cosine_scores_docs_all: for each document, the pairwise cosine similarity between the construct and document token/s
# 	"""
# 	# Create a dictionary of embeddings for all construct tokens (each construct can be 1 or a whole lexicon)
# 	constructs = construct_tokens_d.keys()
# 	construct_embeddings_all = {}
# 	for construct in constructs:
# 		if method.startswith("word"):
# 			construct_tokens = construct_tokens_d.get(construct)
# 			construct_embeddings = construct_embeddings_d.get(construct_tokens)
# 		elif method.startswith("lexicon"):
# 			construct_embeddings = get_construct_embeddings(construct,construct_tokens_d, construct_embeddings_d)
# 		elif method.startswith("avglexicon"):
# 			construct_embeddings = get_construct_embeddings(construct,construct_tokens_d, construct_embeddings_d)
# 			construct_embeddings = np.mean(construct_embeddings, axis=0)
# 		# TODO: could add weighted by similarity to main construct label or by weights provided by user
# 		construct_embeddings = np.array(construct_embeddings, dtype=float)
# 		construct_embeddings_all[construct] = construct_embeddings
# 		# TODO: check if there are any None or 0 embeddings
	
# 	feature_vectors_all = []
	
# 	cosine_scores_docs_all = {}

# 	print(f'computing similarity between {len(constructs)} and {len(docs_embeddings_d.keys())} documents...')
# 	for doc_id in tqdm.tqdm(docs_embeddings_d.keys(), position=0 ):
# 		doc_token_embeddings_i = docs_embeddings_d.get(doc_id) # embeddings for a document
# 		doc_token_embeddings_i = np.array(doc_token_embeddings_i, dtype=float) 

# 		if skip_nan and doc_token_embeddings_i is None:
# 			continue
# 		feature_vectors_doc = [doc_id]
# 		feature_vectors_doc_col_names = [doc_id_col_name]

# 		# compute cosine similarity between each construct and this document
# 		for construct in constructs:
# 			construct_embeddings = construct_embeddings_all.get(construct) # embeddings for a construct	
			
# 			# cosine similarity
# 			if method.startswith("word_"):
# 				assert len(construct_embeddings.shape) == 1
# 				if doc_token_embeddings_i.shape[0] == 0:  # happens when there is an empty str
# 					doc_token_embeddings_i = [np.zeros(construct_embeddings.shape[0])]
# 				cosine_scores_docs_i = cosine_similarity([construct_embeddings], doc_token_embeddings_i)
# 			else:  
# 				# construct is a list of lists
# 				try:
# 					cosine_scores_docs_i = cosine_similarity(construct_embeddings, doc_token_embeddings_i)
# 				except:
# 					print("broke for the following doc ID (returning cosine_similarity 0):")
# 					print(doc_id)
# 					print()
# 					cosine_scores_docs_i = [0]

# 			cosine_scores_docs_all[str(doc_id) + "_" + construct] = cosine_scores_docs_i # save cosine similarities between construct and document tokens

# 			# all summary stats for a single construct will be concatenated side by side
# 			summary_stats_doc_i = []
# 			summary_stats_name_doc_i = []

# 			# todo: obtain construct token ID and doc token ID for top K max. will be different if list of lists
# 			for stat in summary_stat:
# 				function = getattr(np, stat) # e.g. np.max
# 				doc_sim_stat = function(cosine_scores_docs_i)
# 				summary_stats_doc_i.append(doc_sim_stat)
# 				summary_stats_name_doc_i.append(construct + "_"+stat)

# 			# all constructs will be concatenated side by side
# 			feature_vectors_doc.extend(summary_stats_doc_i)
# 			feature_vectors_doc_col_names.extend(summary_stats_name_doc_i)

# 		# create DF for a single document
# 		feature_vectors_doc_df = pd.DataFrame(feature_vectors_doc, index=feature_vectors_doc_col_names).T
# 		feature_vectors_all.append(feature_vectors_doc_df)

# 	feature_vectors_all = pd.concat(feature_vectors_all).reset_index(drop=True) 

# 	# Scale between 0 and 1 to follow output range of other classification models.
# 	if minmaxscaler is not None:
# 		scaler = MinMaxScaler()
# 		feature_cols = [col for col in feature_vectors_all.columns if any(string in col for string in summary_stat)]
# 		feature_vectors_all[feature_cols] = scaler.fit_transform(feature_vectors_all[feature_cols].values)
# 	if return_cosine_similarity:
# 		return feature_vectors_all, cosine_scores_docs_all
# 	else:
# 		return feature_vectors_all
