# """
# Author: Daniel Low
# License: Apache 2.0
# """

import datetime
import pickle
import dill
import os
import numpy as np
import pandas as pd
import tqdm
import concurrent.futures
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from utils import spacy_tokenizer


# def get_construct_embeddings_as_list(construct, lexicon_dict, construct_embeddings_d):
#     """
    

#     Args:
#         construct (str): The construct to retrieve embeddings for.
#         lexicon_dict (dict): A dictionary mapping construct names to lists of tokens.
#         construct_embeddings_d (dict): A dictionary mapping tokens to their corresponding embeddings.

#     Returns:
#         list: A list of embeddings for the given construct.

#     Raises:
#         AssertionError: If the construct is not a list of prototypes.
#     """
#     # Retrieve the tokens for the given construct from the dictionary
#     lexicon_tokens = lexicon_dict.get(construct)
#     if not isinstance(lexicon_tokens, list):
#         # Raise an error if the construct is not a list of prototypes
#         raise AssertionError("to run method lexicon_*, the construct should be a list of prototypes")
    
#     # Initialize a list to store the embeddings for the given construct
#     construct_embeddings_list = []
    
#     # Iterate over the tokens in the construct
#     for token in lexicon_tokens:
#         # Retrieve the embedding for the current token from the dictionary
#         token_embedding = construct_embeddings_d.get(token)
        
#         # If the embedding is None, print a message and continue to the next token
#         if str(token_embedding) == "None":
#             print("(you should FIX) Could not retrieve embedding for: ", token)
#         else:
#             # Append the embedding to the list of embeddings for the construct
#             construct_embeddings_list.append(token_embedding)
    
#     # Return the list of embeddings for the given construct
#     return construct_embeddings_list

def process_document(doc_id, docs_embeddings_d, construct_embeddings_all, constructs, construct_representation, summary_stat, skip_nan, doc_id_col_name):
	"""
	Process a document and compute cosine similarities between the document and each construct.

	Args:
		doc_id (str): The ID of the document.
		docs_embeddings_d (dict): A dictionary mapping document IDs to their embeddings.
		construct_embeddings_all (dict): A dictionary mapping construct names to their embeddings.
		constructs (list): A list of construct names.
		construct_representation (str): The representation of the constructs.
		summary_stat (list): A list of summary statistics to compute.
		skip_nan (bool): Whether to skip documents with no embeddings.
		doc_id_col_name (str): The name of the column for the document ID.

	Returns:
		tuple: A tuple containing the feature vectors for the document and the cosine scores for each construct.
			- feature_vectors_doc_df (pd.DataFrame): The feature vectors for the document.
			- cosine_scores_docs_all (dict): The cosine scores for each construct.
	"""
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
		if construct_representation.startswith("word_"):
			assert len(construct_embeddings.shape) == 1
			if doc_token_embeddings_i.shape[0] == 0:  # happens when there is an empty str
				doc_token_embeddings_i = [np.zeros(construct_embeddings.shape[0])]
			cosine_scores_docs_i = cosine_similarity([construct_embeddings], doc_token_embeddings_i)
		else:
			# cosine similarity between embedding of construct and document
			cosine_scores_docs_i = cosine_similarity(construct_embeddings, doc_token_embeddings_i)

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
	lexicon_dict,
	documents,
	construct_representation="lexicon",  
	document_representation ='clause',  
	summary_stat=["max"],  
	minmaxscaler=(0, 1),  
	return_cosine_similarity=True,  
	embeddings_model = 'all-MiniLM-L6-v2',  
	document_embeddings_path = './',  
	save_partial_embeddings = True,  
	stored_embeddings_path = None,  
	skip_nan=False,  
	remove_stat_name_from_col_name=False,  
	doc_id_col_name='doc_id'  
):	
	"""
	Measure the similarity between constructs and documents.

	Args:
		lexicon_dict: (dict) mapping construct names to lists of tokens
		documents: (list) list of strings
		construct_representation: (str) how to represent constructs. Possible values: "lexicon", "word_lexicon", "avg_lexicon", "weighted_avg_lexicon"
		document_representation: (str) how to represent documents. Possible values: "unigram", "clause", "sentence", "document"
		summary_stat: (list) list of summary statistics to compute. Possible values: "max", "min", "mean", "sum", "std"
		minmaxscaler: (tuple) range to scale summary statistics. Possible values: (int, int) or None
		return_cosine_similarity: (bool) whether to return cosine similarity. Possible values: True or False
		embeddings_model: (str) name of sentence embeddings model. Possible values: see "Models" here: https://huggingface.co/sentence-transformers and here (click All models upper right corner of table): https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
		document_embeddings_path: (str) path to store document embeddings. Possible values: file path
		save_partial_embeddings: (bool) whether to save partial document embeddings. 
		stored_embeddings_path: (str) path to pickle of stored embeddings. 
		skip_nan: (bool) whether to skip documents with no embeddings. 
		remove_stat_name_from_col_name: (bool) whether to remove summary stat name from column name. 
		doc_id_col_name: (str) name of doc_id column. 

	Returns:
		tuple: A tuple containing the feature vectors for the document and the cosine scores for each construct.
	"""
	
	# Tokenize documents
	if construct_representation == 'document':
		docs_tokenized = documents.copy()
	else:
		# TODO: add arguments as measure() arguments using kwargs.
		docs_tokenized = spacy_tokenizer(documents, 
										language = 'en', model='en_core_web_sm', 
										method = document_representation,
										lowercase=False, 
										display_tree = False, 
										remove_punct=False, 
										clause_remove_conj = True)
	
	
	# Embed construct 
	# ================================================================================================
	# Concatenate all tokens so we don't vectorize the same token multiple times
	lexicon_tokens_concat= lexicon_dict.values()
	lexicon_tokens_concat = list(set(lexicon_tokens_concat))

	if stored_embeddings_path is not None:
		stored_embeddings = dill.load(open(stored_embeddings_path, "rb"))
	
	# If you need to encode new tokens:
	tokens_to_encode = [n for n in lexicon_tokens_concat if n not in stored_embeddings.keys()]
	sentence_embedding_model = SentenceTransformer(embeddings_model)       # load embedding
	
	print("Default input sequence length:", sentence_embedding_model.max_seq_length) 
	tokens_to_encode = [n for n in lexicon_tokens_concat if n not in stored_embeddings.keys()]
			
	# Encode new tokens
	if tokens_to_encode != []:
		embeddings = sentence_embedding_model.encode(tokens_to_encode, convert_to_tensor=True,show_progress_bar=True)	
		embeddings_d = dict(zip(tokens_to_encode, embeddings))
		stored_embeddings.update(embeddings_d)
		# save pickle of embeddings
		with open(stored_embeddings_path, 'wb') as handle:
			dill.dump(stored_embeddings, handle, protocol=dill.HIGHEST_PROTOCOL)

	construct_embeddings_d = {}

	for construct, tokens in lexicon_dict.items():
		construct_embeddings_d[construct] = []
		for token in tokens:
			construct_embeddings_d[construct].append(stored_embeddings.get(token))

	
	# Embed documents
	# ================================================================================================
	# 100m 6000 long conversations with interaction
	ts = datetime.datetime.utcnow().strftime('%y-%m-%dT%H-%M-%S')
	docs_embeddings_d = {}
	i_str_all = []
	for i, list_of_clauses in enumerate(docs_tokenized):
		docs_embeddings_d[i] = sentence_embedding_model.encode(list_of_clauses, convert_to_tensor=True,show_progress_bar=False)	

		if save_partial_embeddings and i%500==0:
			i_str = str(i).zfill(5)
			i_str_all.append(i_str)
			with open(document_embeddings_path+f'embeddings_{embeddings_model}_docs_{document_representation}_with-interaction_{ts}_part-{i_str}.pickle', 'wb') as handle:
				pickle.dump(docs_embeddings_d, handle, protocol=pickle.HIGHEST_PROTOCOL)

	# save final one
	with open('./data/input/ctl/embeddings/'+f'embeddings_{embeddings_model}_docs_tokenized_with-interaction_{ts}.pickle', 'wb') as handle:
		pickle.dump(docs_embeddings_d, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
	# TODO: erase prior ones:
	for i_str in i_str_all:
		os.remove(document_embeddings_path+f'embeddings_{embeddings_model}_docs_{document_representation}_with-interaction_{ts}_part-{i_str}.pickle')

	
		
	construct_embeddings_all = {}
	constructs = lexicon_dict.keys()
	
	for construct in constructs:
		# construct_embeddings_list = get_construct_embeddings_as_list(construct, lexicon_dict, construct_embeddings_d)
		if construct_representation == "avg_lexicon":
			construct_embeddings_list = construct_embeddings_d.get(construct)
			construct_embeddings_list = np.mean(construct_embeddings_list, axis=0)
		# # TODO: 
		# elif











# optimized but not faster
# ===================================
	


# import concurrent.futures
# import numpy as np
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.preprocessing import MinMaxScaler

# def get_construct_embeddings(construct, lexicon_dict, construct_embeddings_d):
# 	lexicon_tokens = lexicon_dict.get(construct)
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
# 	lexicon_dict=None,
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
# 	constructs = lexicon_dict.keys()
# 	for construct in constructs:
# 		construct_embeddings = get_construct_embeddings(construct, lexicon_dict, construct_embeddings_d)
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

# def get_construct_embeddings(construct,lexicon_dict, construct_embeddings_d):
# 	lexicon_tokens = lexicon_dict.get(construct)

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
# 	lexicon_dict=None,
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
# 			lexicon_dict: 
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
# 	constructs = lexicon_dict.keys()
# 	construct_embeddings_all = {}
# 	for construct in constructs:
# 		if method.startswith("word"):
# 			construct_tokens = lexicon_dict.get(construct)
# 			construct_embeddings = construct_embeddings_d.get(construct_tokens)
# 		elif method.startswith("lexicon"):
# 			construct_embeddings = get_construct_embeddings(construct,lexicon_dict, construct_embeddings_d)
# 		elif method.startswith("avglexicon"):
# 			construct_embeddings = get_construct_embeddings(construct,lexicon_dict, construct_embeddings_d)
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
