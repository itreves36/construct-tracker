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
from construct_tracker.utils.tokenizer import spacy_tokenizer
import numpy as np
from IPython.display import HTML, display
import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")


# Local
from construct_tracker import lexicon
from .utils.logger_config import setup_logger

# Set up the logger
logger = setup_logger()


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
			- cosine_scores_docs_all (dict): The cosine scores for each construct (columns are document tokens, rows are lexicon tokens).
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
	count_if_exact_match = 'sum',
	lexicon_counts = None,
	similarity_threshold = 0.30,
	minmaxscaler=None,  
	return_cosine_similarity=True,  
	embeddings_model = 'all-MiniLM-L6-v2',  
	document_embeddings_path = './data/embeddings/',  
	save_lexicon_embeddings = False,
	save_doc_embeddings = False,
	save_partial_doc_embeddings = True,  
	stored_embeddings_path = None,  
	skip_nan=False,  
	remove_stat_name_from_col_name=False,  
	doc_id_col_name='document_id',
	verbose=True  
):	
	"""
	Measure the similarity between constructs and documents.

	Args:
		lexicon_dict: (dict) mapping construct names to lists of tokens
		documents: (list) list of strings
		construct_representation: (str) how to represent constructs. Possible values: "lexicon", "word_lexicon", "avg_lexicon", "weighted_avg_lexicon"
		document_representation: (str) how to represent documents. Possible values: "unigram", "clause", "sentence", "document"
		summary_stat: (list) list of summary statistics to compute. Possible values: "max", "min", "mean", "sum", "std"
		count_if_exact_match: (False, 'sum', 'replace') if a document contains a lexicon token, 'replace'returns count (positive int replaces cosine similarity), False returns cosine similarity to find similar tokens that are not in lexicon, 'sum' returns the sum of counts and cosine similarity. Default: 'sum' (to obtain a bit information from both)
		lexicon_counts: (None or counts pandas dataframe): only used if count_if_exact_match is True; if you want to compute counts in a specific way use your_lexicon.extract() and input the counts dataframe here. 
		similarity_threshold: (None, float) avoid using very low cosine similarities by setting a threshold below which to replace with NaN. Should be tested given it will depend on the embeddings. Recommended possible values: None (do not apply) or float [0.3,0.7], will interact with count_if_exact_match argument. 
		minmaxscaler: (tuple or False) range to scale summary statistics. Possible values: (int, int) or None
		return_cosine_similarity: (bool) whether to return cosine similarity. Can occupy a lot of memory if you have many documents and many tokens per document. Possible values: True or False
		embeddings_model: (str) name of sentence embeddings model. Possible values: see "Models" here: https://huggingface.co/sentence-transformers and here (click All models upper right corner of table): https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
		document_embeddings_path: (str) path to store document embeddings. Possible values: file path
		save_embeddings: (bool) whether to save document embeddings. 
		save_partial_doc_embeddings: (bool) whether to save partial document embeddings. 
		stored_embeddings_path: (str) path to pickle of stored embeddings. 
		skip_nan: (bool) whether to skip documents with no embeddings. 
		remove_stat_name_from_col_name: (bool) whether to remove summary stat name from column name. 
		doc_id_col_name: (str) name of doc_id column. 
		verbose: (bool) whether to print progress (True) or just warnings (False).

	Returns:
		tuple: A tuple containing the feature vectors for the document and the cosine scores for each construct.
	"""

	if verbose:
		logger.setLevel(logging.DEBUG)
	else:
		logger.setLevel(logging.WARNING)
	
	
	# Make sure summary_stat is a single statistic, and recommedn using ['max']
	if count_if_exact_match:
		try: assert len(summary_stat) == 1
		except AssertionError as e:
			raise ValueError("when count_if_exact_match is different than False, summary_stat should be a single statistic like ['max'], because we'll replace cosine similarities with counts >=1.")
		
		try: assert summary_stat[0] == 'max'
		except:
			logging.warning("when count_if_exact_match is True, we recommend summary_stat be ['max'] to max cosine similarity (approximate semantic match up to value of 1) with counts (definite exact match of value 1 or higher). If you replaced mean or std cosine similarities with counts, this is harder to interpret how similarities and counts are related.")

	# Embed construct: construct_embeddings_d
	# ================================================================================================
	# Concatenate all tokens so we don't vectorize the same token multiple times
	lexicon_tokens_concat = [item for sublist in lexicon_dict.values() for item in sublist]

	
	if stored_embeddings_path is not None:
		logger.info("Loading existing lexicon token embeddings")	
		stored_embeddings = dill.load(open(stored_embeddings_path, "rb"))
		# If you need to encode new tokens:
		tokens_to_encode = [n for n in lexicon_tokens_concat if n not in stored_embeddings.keys()]

	else:
		logger.warning("stored_embeddings_path is None. Extracting all lexicon token embeddings from scratch instead of loading stored embeddings...")	
		stored_embeddings = {}
		stored_embeddings_path = 'lexicon_embeddings.pickle'


	sentence_embedding_model = SentenceTransformer(embeddings_model)       # load embedding
	
	logger.info(f"Default input sequence length for {embeddings_model}: {sentence_embedding_model.max_seq_length}") 

	tokens_to_encode = [n for n in lexicon_tokens_concat if n not in stored_embeddings.keys()]
			
	# Encode new tokens
	if tokens_to_encode != []:
		logger.info(f"Encoding {len(tokens_to_encode)} new construct tokens...")
		embeddings = sentence_embedding_model.encode(tokens_to_encode, convert_to_tensor=True,show_progress_bar=True)	
		embeddings_d = dict(zip(tokens_to_encode, embeddings))
		stored_embeddings.update(embeddings_d)
		logger.info(f"Finished encoding construct tokens.")
		# save pickle of embeddings
		if save_lexicon_embeddings:
			logger.info(f"Saving lexicon token embeddings here: {stored_embeddings_path}")
			with open(stored_embeddings_path, 'wb') as handle:
				dill.dump(stored_embeddings, handle, protocol=dill.HIGHEST_PROTOCOL)

	construct_embeddings_d = {}
	lexicon_dict_final_order = {}
	for construct, tokens in lexicon_dict.items():
		construct_embeddings_d[construct] = []
		lexicon_dict_final_order[construct] = []
		for token in tokens:
			construct_embeddings_d[construct].append(stored_embeddings.get(token))
			lexicon_dict_final_order[construct].append(token)
	
	# Average embeddings for a single construct
	constructs = lexicon_dict.keys()
	if construct_representation == "avg_lexicon":
		for construct in constructs:
			construct_embeddings_list = construct_embeddings_d.get(construct)
			construct_embeddings_avg = np.mean(construct_embeddings_list, axis=0)
			construct_embeddings_avg = np.array(construct_embeddings_avg, dtype=float)
			construct_embeddings_d[construct] = construct_embeddings_avg
	# # TODO:
	# elif construct_representation == "weighted_avg_lexicon": 
	
	# Embed documents: docs_embeddings_d
	# ================================================================================================
	# 100m 6000 long conversations with interaction

	# Tokenize documents
	if construct_representation == 'document':
		docs_tokenized = documents.copy()
	else:
		logger.info("Tokenizing documents...")
		# TODO: add arguments as measure() arguments using kwargs.
		
		docs_tokenized = spacy_tokenizer(documents, 
										language = 'en', model='en_core_web_sm', 
										method = document_representation,
										lowercase=False, 
										display_tree = False, 
										remove_punct=False, 
										clause_remove_conj = True)
		logger.info("Finished tokenization.")

	ts = datetime.datetime.utcnow().strftime('%y-%m-%dT%H-%M-%S')
	docs_embeddings_d = {}
	i_str_all = []
	logger.info("Encoding all document tokens...")
	
	
	if save_doc_embeddings:
		# todo add ts
		document_embeddings_path = document_embeddings_path+f'_{ts}/'
		os.makedirs(document_embeddings_path, exist_ok=True)
	for i, list_of_clauses in enumerate(docs_tokenized):
		docs_embeddings_d[i] = sentence_embedding_model.encode(list_of_clauses, convert_to_tensor=True,show_progress_bar=False)	

		if save_doc_embeddings and save_partial_doc_embeddings and i%500==0:
			# save partial ones in case it fails during the process
			i_str = str(i).zfill(5)
			i_str_all.append(i_str)
			with open(document_embeddings_path+f'embeddings_{embeddings_model}_docs_{document_representation}_with-interaction_{ts}_part-{i_str}.pickle', 'wb') as handle:
				pickle.dump(docs_embeddings_d, handle, protocol=pickle.HIGHEST_PROTOCOL)

	# save final one
	if save_doc_embeddings:
		logger.info("Saving document embeddings here: "+document_embeddings_path+f'embeddings_{embeddings_model}_docs_{document_representation}_with-interaction_{ts}.pickle')
		with open(document_embeddings_path+f'embeddings_{embeddings_model}_docs_{document_representation}_with-interaction_{ts}.pickle', 'wb') as handle:
			pickle.dump(docs_embeddings_d, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
	# remove partial ones
	if save_doc_embeddings and save_partial_doc_embeddings:
		for i_str in i_str_all:
			os.remove(document_embeddings_path+f'embeddings_{embeddings_model}_docs_{document_representation}_with-interaction_{ts}_part-{i_str}.pickle')

	
	# Compute cosine similarity
	# ================================================================================================
	feature_vectors_all = []
	cosine_scores_docs_all = {}

	logger.info(f'computing similarity between {len(constructs)} constructs and {len(docs_embeddings_d.keys())} documents...')

	# parallelized processing
	with concurrent.futures.ThreadPoolExecutor() as executor:
		futures = [executor.submit(process_document, doc_id, docs_embeddings_d, construct_embeddings_d, constructs, construct_representation, summary_stat, skip_nan, doc_id_col_name) for doc_id in docs_embeddings_d.keys()]
		for future in tqdm.tqdm(concurrent.futures.as_completed(futures)):
			doc_result, doc_cosine_scores = future.result()
			if doc_result is not None:
				feature_vectors_all.append(doc_result)
				cosine_scores_docs_all.update(doc_cosine_scores)
				

	
	feature_vectors_all = pd.concat(feature_vectors_all).reset_index(drop=True) 
	
	logger.info('Finished.')

		# Scale between 0 and 1 to follow output range of other classification models.
	if minmaxscaler is not None:
		scaler = MinMaxScaler()
		feature_cols = [col for col in feature_vectors_all.columns if any(string in col for string in summary_stat)]
		feature_vectors_all[feature_cols] = scaler.fit_transform(feature_vectors_all[feature_cols].values)

	if remove_stat_name_from_col_name:
		for stat in summary_stat:
			feature_vectors_all.columns = [n.replace(f'_{stat}', '') for n in feature_vectors_all.columns]

	
	if similarity_threshold is not None:
		# Set all values below threshold to 0
		# feature_vectors_all = feature_vectors_all[counts.columns]
		feature_vectors_all[feature_vectors_all < similarity_threshold] = 0

	# Count lexicon in documents
	# =================================================================================================
	if count_if_exact_match:
		if not isinstance(lexicon_counts, pd.DataFrame):
			# if you did not provide the counts DF, lexicon_counts will be None, and it will be generated here 
			new_lexicon = lexicon.Lexicon()
			for c in lexicon_dict.keys():
				new_lexicon.add(c, section = 'tokens', value = lexicon_dict[c], verbose = False)
		
			counts, matches_by_construct, matches_doc2construct, matches_construct2doc  = new_lexicon.extract(documents,
																								normalize = False,
																								)

		counts = counts[list(lexicon_dict.keys())]
		if not remove_stat_name_from_col_name:
			counts.columns = [n+f'_{summary_stat[0]}' for n in counts.columns]
		
		# display(counts)
		# display(feature_vectors_all)
		if count_if_exact_match == 'sum':
			feature_vectors_all[counts.columns] = feature_vectors_all[counts.columns]+counts
			
		elif count_if_exact_match == 'replace':
			feature_vectors_all = counts.where(counts >= 1, feature_vectors_all)
	
	# feature_vectors_all[doc_id_col_name] = range(len(feature_vectors_all))

	

	
	
	# add documents
	construct_columns = list(feature_vectors_all.columns[1:])
	feature_vectors_all['documents'] = documents 
	feature_vectors_all['documents_tokenized'] = docs_tokenized
	# reorder
	feature_vectors_all = feature_vectors_all[[doc_id_col_name, 'documents', 'documents_tokenized'] +construct_columns]
	# feature_vectors_all.sort_values(by=[doc_id_col_name], inplace=True)
	feature_vectors_all.reset_index(drop=True, inplace=True)


	
	if return_cosine_similarity:
		return feature_vectors_all, docs_tokenized, lexicon_dict_final_order, cosine_scores_docs_all
	else:
		return feature_vectors_all
	


def get_highest_similarity_phrase(doc_id, construct, documents, documents_tokenized, cosine_similarities, lexicon_dict_final_order):
	# You can verify similarity here: https://github.com/danielmlow/tutorials/blob/main/text/semantic_similarity.ipynb 
	doc_id_construct = f'{doc_id}_{construct}'
	similarities_for_1_doc = cosine_similarities[doc_id_construct]
	most_similar_lexicon_token_per_doc_token_index = np.argmax(similarities_for_1_doc, axis=0) # top similarities for each token in the document
	highest_similarity = np.max(similarities_for_1_doc)
	highest_similarity = np.round(highest_similarity,2)
	most_similar_document_token_index = np.argmax(np.max(similarities_for_1_doc, axis=0))
	most_similar_lexicon_token_index = most_similar_lexicon_token_per_doc_token_index[most_similar_document_token_index]
	most_similar_document_token = documents_tokenized[doc_id][most_similar_document_token_index]
	most_similar_lexicon_token = lexicon_dict_final_order.get(construct)[most_similar_lexicon_token_index]

	# Display
	print(f"The construct '{construct}' through its token '{most_similar_lexicon_token}' had the highest cosine similarity ({highest_similarity}) with the following document token:\n'{most_similar_document_token}'")
	document = documents[doc_id]
	document = document.replace(most_similar_document_token, f"<mark>{most_similar_document_token}</mark>") 
	display(HTML(document))

	return most_similar_lexicon_token, most_similar_document_token, highest_similarity











"""
documents = ['He is too competitive',
 'Every time I speak with my cousin Bob, I have great moments of insight, clarity, and wisdom',
 "He meditates a lot, but he's not super smart"]

tokens = [
	['insight', 'clarity', 'realization'],
	['mindfulness', 'meditation', 'buddhism'],
	['bad', 'mean', 'crazy'],
	]


label_names = ['insight', 'mindfulness', 'bad']


lexicon_dict = dict(zip(label_names, tokens))
lexicon_dict

feature_vectors_lc, cosine_scores_docs_lc = measure(
	lexicon_dict ,
	documents,
	)
"""


