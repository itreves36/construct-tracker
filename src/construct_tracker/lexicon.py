import datetime
import json
import re
import sys
import time
import warnings
from collections import Counter

import dill
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append("/Users/danielmlow/Dropbox (MIT)/datum/construct-tracker/")
from construct_tracker.genai import api_request  # local
from construct_tracker.utils import lemmatizer  # local / construct-tracker package
from construct_tracker.utils.word_count import word_count  # local

# # Timeout handler function
# def timeout_handler(signum, frame):
# 	raise TimeoutError()


def generate_variable_name(str):
	"""
	Replace spaces with undersore, lower-case, remove certain punctuation
	:param str:
	:return:
	"""
	variable_name = (
		str.replace(",", "").replace(" & ", "_").replace(" and ", "_").replace(" ", "_").replace("-", "_").lower()
	)
	return variable_name


# TODO: set prompt with template for construct so we don't need to set it before every construct
def generate_prompt(
	construct,
	prompt_name=None,
	prompt="default",
	domain=None,
	definition=None,
	examples=None,
	output_format="default",
	remove_parentheses_definition=True,
):
	if output_format == "default":
		output_format = (
			"Each token should be separated by a semicolon. Do not return duplicate tokens. Do not provide any"
			" explanation or additional text beyond the tokens."
		)
	# removed: Order them by how similar they are to {construct}.
	elif output_format == "json":
		output_format = (
			"Provide tokens in JSON output. Do not return duplicate tokens. Do not provide any explanation or"
			" additional text beyond the tokens."
		)

	# Prompt
	if not isinstance(prompt_name, str):
		# if prompt_name == None:
		prompt_name = construct.replace("_", " ").lower()

	if prompt == "default":
		prompt = "Provide many single words and some short phrases related to"
		if domain:
			domain = f"(in the {domain} domain). "
			prompt = f"""{prompt} {prompt_name} {domain}{output_format}"""
		else:
			prompt = f"""{prompt} {prompt_name}. {output_format}"""
		if definition:
			if remove_parentheses_definition:
				definition = re.sub(r"\(.*?\)", "", definition)  # remove parentheses which is usually citation.
			prompt += f"\nHere is a definition of {prompt_name}: {definition.lower().strip()}"

		if isinstance(examples, list):
			examples = "; ".join(examples)
		if isinstance(examples, str):
			# examples = '; '.join(examples)
			prompt += f"\nHere are some examples (include these in the list): {examples}."

	return prompt


# for var_name, definition, definition_references, clean_name, examples in definitions_df.values:
#
# 	with open('./data/lexicons/suicide_risk_constructs_and_definitions.txt', 'a+') as f:
# 		f.write(f'{clean_name.capitalize()}\n')
# 		f.write(f'- Definition: {definition}\n')
# 		f.write(f'- Examples: {examples}\n')
# 		f.write(f'- Sources: {definition_references}\n')
# 		f.write(f'---------------------------------------------------------------------------\n\n')


# Set the signal handler for timeout
# signal.signal(signal.SIGALRM, timeout_handler)
# try:
#     signal.alarm(timeout)  # Start the timer
#     responses = completion(model=model, messages=messages, api_key=api_key, temperature=temperature)
#     response = responses.get("choices")[0].get("message").get("content")  # access response for first message
#     return response
# except TimeoutError:
#     print(f"Error: The code has exceeded the time limit ({timeout} seconds) and has been stopped.")
#     return


def find_partial_matching_strings(list_a, list_b):
	"""
	Finds strings in list_a that contain any of the strings in list_b,
	but does not return the string if it is identical to an element in list_b.

	Parameters:
	list_a (list of str): The list of strings to search within.
	list_b (list of str): The list of substrings to search for.

	Returns:
	list of str: A list containing strings from list_a that have any of the substrings from list_b,
																									 but are not identical to any string in list_b.
	"""

	partial_matching_strings = []
	matched_substrings = {}
	for string_a in list_a:
		for string_b in list_b:
			if string_b in string_a and string_a != string_b:
				partial_matching_strings.append(string_a)
				matched_substrings[string_a] = string_b
	return partial_matching_strings, matched_substrings


def count_lexicons_in_doc(doc, tokens=[], return_zero=[], return_matches=False):
	# TODO, perhaps return match and context (3 words before and after)
	"""

	Args:
																	doc:
																	tokens: lexicon tokens
																	return_zero:
																	normalize:
																	return_matches:

	Returns:

	"""

	text = re.sub(
		"[^\\w\\d'\\s]+", "", doc.lower()
	)  # remove punctuation except apostrophes because we need to search for things like "don't want to live"
	counter = 0
	matched_tokens = []
	for token in tokens:
		token = token.lower()
		matches = text.count(token)
		counter += matches
		if return_matches and matches > 0:
			matched_tokens.append(token)
	if return_matches:
		return counter, matched_tokens
	else:
		return counter


def analyze(docs, lexicon_token_d, normalize=True, return_zero=[], return_matches=False, add_word_count=True):
	# TODO: return zero is for entire docs, shouldnt it be for tokens?
	"""

	Args:
																	docs:
																	lexicons:
																	normalize:
																																	divide by zero
																	return_zero:

	Returns:

	"""
	# process all posts
	# docs is list of list
	# lexicons is dictionary {'construct':[token1, token2, ...], 'construct2':[]}
	docs = [doc.replace("\n", " ").replace("  ", " ").replace("“", "").replace("”", "") for doc in docs]
	feature_vectors = {}
	matches = {}

	for construct in list(lexicon_token_d.keys()):
		lexicon_tokens = lexicon_token_d.get(construct)
		if return_matches:
			counts_and_matched_tokens = [
				count_lexicons_in_doc(
					doc, tokens=lexicon_tokens, return_zero=return_zero, return_matches=return_matches
				)
				for doc in docs
			]
			counts = [n[0] for n in counts_and_matched_tokens]
			matched_tokens = [n[1] for n in counts_and_matched_tokens if n[1] != []]
			matches[construct] = matched_tokens

		else:
			counts = [
				count_lexicons_in_doc(
					doc, tokens=lexicon_tokens, return_zero=return_zero, return_matches=return_matches
				)
				for doc in docs
			]
		# one_construct = one_construct/word_counts #normalize

		feature_vectors[construct] = counts

		# # feature_vector = extract_NLP_features(post, features) #removed feature_names from output
		# if len(feature_vector) != 0:
		#      raw_series = list(df_subreddit.iloc[pi])
		#      subreddit_features = subreddit_features.append(pd.Series(raw_series + feature_vector, index=full_column_names), ignore_index=True)

	# feature_vectors0   = pd.DataFrame(docs, columns = ['docs'])
	# feature_vectors = pd.concat([feature_vectors0,pd.DataFrame(feature_vectors)],axis=1)
	feature_vectors = pd.DataFrame(feature_vectors)

	#      feature_vectors   = pd.DataFrame(docs)
	#      feature_vectors['docs']=docs

	if normalize:
		wc = word_count(docs, return_zero=return_zero)
		wc = np.array(wc)
		feature_vectors_normalized = np.divide(feature_vectors.values.T, wc).T
		feature_vectors = pd.DataFrame(
			feature_vectors_normalized, index=feature_vectors.index, columns=feature_vectors.columns
		)

	if add_word_count and normalize:
		feature_vectors["word_count"] = wc
	elif add_word_count and not normalize:
		wc = word_count(docs, return_zero=return_zero)
		feature_vectors["word_count"] = wc

		# feature_vectors = feature_vectors/wc

	if return_matches:
		# all lexicons
		matches_counter_d = {}
		for name_i in list(lexicon_token_d.keys()):
			if matches.get(name_i):
				x = Counter([n for i in matches.get(name_i) for n in i])
				matches_counter_d[name_i] = matches_counter_d[name_i] = {
					k: v for k, v in sorted(x.items(), key=lambda item: item[1])
				}
		# Counter([n for i in matches.get(name_i) for n in i]) for name_i in lexicons_d.keys()]

		return feature_vectors, matches_counter_d
	else:
		return feature_vectors


def remove_substrings(s, substrings):
	for substring in substrings:
		s = s.replace(substring, "")
	return s


class Lexicon:
	def __init__(self, name=None, description=None):
		"""
		Initializes the class with optional name and description parameters.
		"""
		self.name = name
		self.description = description
		self.constructs = {}
		self.construct_names = []
		if isinstance(name, str):
			# Load the lexicon if name is a string
			self.constructs = load_lexicon(name)
		self.exact_match_n = 4
		self.exact_match_tokens = []
		self.remove_from_all_constructs = []
		self.attributes = {}  # for any other attribute to be added dynamically

		return

	def set_attribute(self, key, value):
		self.attributes[key] = value

	def get_attribute(self, key, default=None):
		return self.attributes.get(key, default)

	def analyze(self, documents, normalize=True, return_matches=False):
		"""
		normalize: if True, divide by word count (generally a good idea, an appearance of loneliness in a short doc should weigh more than in a long doc)
		"""
		analysis_d = {}

		return analysis_d

	def generate_definition(self, construct, domain=None, timeout=45, num_retries=2):
		""" """
		if domain:
			prompt = (  # perhaps add: 'only what is is, not how it affects other things.""
				f"Provide a very brief definition of {construct} (in the {domain} domain)."
			)
		else:
			prompt = (  # perhaps add: 'only what is is, not how it affects other things.""
				f"Provide a very brief definition of {construct}."
			)
		definition = api_request(prompt, timeout=timeout, num_retries=num_retries)

		return definition

	def clean_response(self, response, type="tokens"):
		# print('0a', response)
		# response = "Here is a list of tokens related to sexual abuse and harassment, separated by semicolons: violated; abused; assaulted; raped; molested; harasses; nonconsensual; harassment; victimized; stalking; groping; coerced; profaned; derided; violated my boundaries. "
		# response = 'Acceptance and Commitment Therapy (ACT)'
		if type == "tokens":
			# print('0b', response)
			# response = lexicon.constructs['sexual_abuse_harassment']['tokens']
			# response = "suicide; kill; death; destroy; suffer; distress; desperate; ceased; expire; hurt; isolate; lonely; agonize; pain; anguish; grief; slit; bleed; hang; overdose; freeze to death; jump; fall; throttle; immolate; lie in traffic; hang oneself; jump off a bridge; jump in front of a train; drug overdose; hanging; drowning; suffocation; bullet wound; glass throat; carbon monoxide; running in front of a train; jumping from a building; steering into oncoming traffic; laying in the middle of the railroad tracks; tying a rope; waking up with a knife next to one's throat; planning; preparing; time; place; method; final; irreversible."
			# if ';' in response:
			response_list = response.split(";")

			tokens = []
			try:
				for t in response_list:
					if t == "":
						continue
					elif isinstance(t, str) and len(t) > 0:
						if "\n" in t:
							t = t.split("\n")
							# print(t)

							t = [n[0].lower() + n[1:] if (not n.isupper()) else n for n in t]
							tokens.extend(t)
						else:
							if t.isupper() or t[:2] == "I " or t[:2] == "I'":
								tokens.append(t)
							else:
								tokens.append(t[0].lower() + t[1:])
					else:
						warnings.warn(f"token is either not str or some other issue: '{t}'. Not adding to lexicon.")
				tokens = [n.strip() for n in tokens]
				tokens_new = []
				for token in tokens:
					if "(" in t and ")" in t:
						parentheses_content = re.findall(r"\((.*?)\)", t)
						tokens_new.extend(parentheses_content)  # extend because it returns a list
						without_parentheses_content = re.sub(r"\(.*?\)", "", t)
						tokens_new.append(without_parentheses_content)  # remove .lower() here
					else:
						tokens_new.append(token)

				tokens = tokens_new.copy()
				remove_if_contains_substrings = ["single words", "here is a list"]
				tokens = [item for item in tokens if all(sub not in item for sub in remove_if_contains_substrings)]

				tokens = [n.strip().strip(".:;,!?") for n in tokens]  # TODO fix
				tokens = [n for n in tokens if n != ""]
				tokens = [
					n.replace("-", " ") if n.count("-") > 1 else n for n in tokens
				]  # for tokens like i-want-to-exit-life
				tokens = list(np.unique(tokens))  # Remove duplicates
				return tokens
			except:
				warnings.warn(
					"Gen AI response could not be parsed. Continuing with the raw response as a single token. This is"
					f" the response: {response}"
				)
				return response_list

		elif type == "definition":
			# TODO
			pass

	def build(self, construct):
		"""
		Union of tokens from each source. Remove tokens that were removed.
		:return: None
		"""

		# add
		tokens = []
		for source in self.constructs[construct]["tokens_metadata"]:
			if self.constructs[construct]["tokens_metadata"][source]["add_or_remove"] == "add":
				tokens_i = self.constructs[construct]["tokens_metadata"][source]["tokens"]
				tokens.extend(tokens_i)
		tokens = list(np.unique(tokens))
		self.constructs[construct]["tokens"] = tokens

		# remove
		remove = []
		for source in self.constructs[construct]["tokens_metadata"]:
			if self.constructs[construct]["tokens_metadata"][source]["add_or_remove"] == "remove":
				tokens_i = self.constructs[construct]["tokens_metadata"][source]["tokens"]
				remove.extend(tokens_i)
		remove = list(np.unique(remove))
		self.constructs[construct]["remove"] = remove

		# If token is in override_remove, it will not be removed
		remove = [n for n in remove if n not in self.constructs[construct]["override_remove"]]

		remove_from_all_constructs = self.get_attribute("remove_from_all_constructs")
		if isinstance(remove_from_all_constructs, list) and len(remove_from_all_constructs) > 0:
			# if self.constructs['config'].get('remove_from_all_constructs')> 0
			# # if getattr(self, 'remove_from_all_constructs', []) > 0:
			self.constructs[construct]["remove"] = list(np.unique(remove + remove_from_all_constructs))

		# Remove tokens that are in remove list
		self.constructs[construct]["tokens"] = [n for n in tokens if n not in remove]

		return

	def add(
		self,
		construct,
		section="tokens",
		value=None,  # str, list or 'create'
		# if value == 'create', do API request with LiteLLM
		prompt=None,  # str, None will create default prompt
		source=None,  # str: model such as 'command-nightly", see litellm for models, or description "manually added by DML". Cohere 'command-nightly' models offer 5 free API calls per minute.
		api_key=None,
		temperature=0.1,
		top_p=1,
		seed=42,
		timeout=120,
		num_retries=2,
		max_tokens=None,
		remove_parentheses_definition=True,  # TODO: remove double spaces
		verbose=True,
	):
		# prompt_name=None,
		# definition=None,
		# definition_references=None,
		# examples = None,
		# tokens=None,
		# tokens_metadata=None,
		self.construct_names.append(construct)
		self.construct_names = list(set(self.construct_names))

		if construct not in self.constructs.keys():
			if verbose:
				warnings.warn(
					f"'{construct}' not in lexicon. Creating new entry for it. This warning is useful so you if you"
					" have a typo in the construct name, you don't add a new entry by mistake."
				)
			self.constructs[construct] = {
				"prompt_name": None,
				"variable_name": generate_variable_name(
					construct
				),  # you can replace it later with lexicon.add(construct, section='variable_name', value=new_name)
				"definition": None,
				"definition_references": None,
				"examples": None,
				"tokens": [],
				"tokens_lemmatized": [],
				"remove": [],
				"override_remove": [],
				"tokens_metadata": {},
			}
		ts = datetime.datetime.utcnow().strftime("%y-%m-%dT%H-%M-%S")  # so you don't overwrite, and save timestamp

		if section == "tokens":
			if isinstance(value, str):
				if value == "create":
					if not isinstance(prompt, str):
						# generate default prompt
						prompt = generate_prompt(
							construct,
							prompt_name=self.constructs[construct]["prompt_name"],
							prompt="default",
							domain=None,
							# domain=self.constructs[construct]['domain'], # need to add domain to construct dict
							definition=self.constructs[construct]["definition"],
							examples=self.constructs[construct]["examples"],
							output_format="default",
							remove_parentheses_definition=remove_parentheses_definition,
						)
					# else, use prompt provided in arguments
					start = time.time()
					response = api_request(
						prompt,
						model=source,
						api_key=api_key,
						temperature=temperature,
						top_p=top_p,
						timeout=timeout,
						num_retries=num_retries,
						max_tokens=max_tokens,
						seed=seed,
					)
					end = time.time()
					time_elapsed = end - start
					time_elapsed = round(time_elapsed, 1)

					tokens = self.clean_response(response, type="tokens")
					# # add examples to tokens
					# if isinstance(examples, list):
					# 	for example in examples:
					# 		if example not in tokens:
					# 			tokens.insert(0, example)
					# 	self.constructs[construct]["tokens"] = tokens
					tokens = list(np.unique(tokens))
					source_info = (
						f"{source}, temperature-{temperature}, top_p-{top_p}, max_tokens-{max_tokens}, seed-{seed},"
						f" {ts}"
					)
					self.constructs[construct]["tokens_metadata"][source_info] = {
						"add_or_remove": "add",
						"tokens": tokens,
						"prompt": prompt,
						"time_elapsed": time_elapsed,
					}
				else:
					raise Exception(
						"value needs to be list of token/s. This operation will not add anything to lexicon. Returning"
						" None."
					)
					return
			elif isinstance(value, list):
				# manually add tokens
				tokens = value.copy()
				source_info = f"{source} {ts}"
				tokens = [n.strip() for n in tokens]
				self.constructs[construct]["tokens_metadata"][source_info] = {"add_or_remove": "add", "tokens": tokens}
			else:
				raise TypeError(
					"value needs to be list of token/s. This operation will not add anything to lexicon. Returning"
					" None."
				)
				return

			# # merge all sources
			# final_tokens = []
			# for source, metadata in self.constructs[construct]['tokens_metadata'].items():
			# 	final_tokens.extend(metadata['tokens'])
			# final_tokens = list(np.unique(final_tokens))
			# self.constructs[construct]['tokens'] = final_tokens
			# if 'remove' in self.constructs[construct]['tokens_metadata'].keys():

			remove_tokens = self.constructs[construct]["remove"]
			try_to_add_but_removed = [n for n in tokens if n in remove_tokens]
			if len(try_to_add_but_removed) > 0:
				warnings.warn(
					f"These tokens are trying to be added to the construct '{construct}' but are listed in the 'remove'"
					f" section of tokens_metadata: {try_to_add_but_removed}.\nThey will only be added to the"
					" tokens_metadata not to the final tokens. You can override the previously removed tokens by"
					f" adding them to the 'override_remove': lexicon.add('{construct}', section='override_remove',"
					" value=tokens_to_be_included). You can also delete any added or removed set of tokens from"
					f" metadata: \ndel lexicon.constructs['{construct}']['tokens_metadata']['source']\nfollowed"
					f" by:\nlexicon.build('{construct})\n"
				)

			# build (add all sources, remove from remove section
			self.build(construct)

		elif section in ["prompt_name", "definition", "definition_references", "examples"]:
			self.constructs[construct][section] = value  # replace value

		return

		# todo: compute percentage of final tokens that are from each source
		# TODO: add tests for all possibilities

	def remove(self, construct, source=None, remove_tokens=None, remove_substrings=None):
		# adds list of tokens to 'remove' section of 'tokens_metadata'.
		ts = datetime.datetime.utcnow().strftime("%y-%m-%dT%H-%M-%S")  # so you don't overwrite, and save timestamp
		if isinstance(source, str):
			source_info = f"{source} {ts}"
		elif source is None:
			source_info = f"{ts}"

		self.constructs[construct]["tokens_metadata"][source_info] = {
			"add_or_remove": "remove",
			"tokens": remove_tokens,
		}

		# TODO: adapt remove_substrings to new code: add in tokens_metada? It's currently not in build.
		if isinstance(remove_substrings, list):
			tokens = self.constructs[construct]["tokens"]
			p = re.compile("|".join(map(re.escape, remove_substrings)))  # escape to handle metachars
			tokens = [p.sub("", s).strip() for s in tokens]
			# Add examples
			for example in self.examples:
				if example not in tokens:
					tokens.insert(0, example)
			# Add construct
			# if construct not in tokens:
			# 	tokens.insert(0, construct)

			tokens = list(np.unique(tokens))
			self.constructs[construct]["tokens"] = tokens
		# self.constructs[construct]["tokens_removed"] =+ (tokens_len - len(tokens))
		self.build(construct)
		return

	def remove_tokens_containing_token(self, construct):
		tokens = self.constructs[construct]["tokens"]
		tokens_len = len(tokens)
		partial_matching_strings, matched_substrings = find_partial_matching_strings(tokens, tokens)
		tokens = [n for n in tokens if n not in partial_matching_strings]  # remove tokens containing tokens
		self.constructs[construct]["tokens"] = tokens
		self.constructs[construct]["tokens_removed"] = +(tokens_len - len(tokens))
		return

	def to_pandas(self, add_annotation_columns=True, add_metadata_rows=True, order=None, tokens="tokens"):
		# def to_pandas(self, add_annotation_columns=True, order=None, tokens = 'tokens'):
		"""
		TODO: still need to test
		lexicon: dictionary with at least
		{'construct 1': {'tokens': list of strings}
		}
		:return: Pandas DF
		"""
		if order:
			warn_missing(self.constructs, order, output_format="pandas / csv")
		lexicon_df = []
		constructs = order.copy() if isinstance(order, list) else self.constructs.keys()
		for construct in constructs:
			df_i = pd.DataFrame(self.constructs[construct][tokens], columns=[construct])
			if add_annotation_columns:
				df_i[construct + "_include"] = [np.nan] * df_i.shape[0]
				df_i[construct + "_add"] = [np.nan] * df_i.shape[0]
			lexicon_df.append(df_i)

		lexicon_df = pd.concat(lexicon_df, axis=1)

		if add_metadata_rows:
			metadata_df_all = []
			if order is None:
				order = self.constructs.keys()

			for construct in order:
				# add definition, examples, prompt_name as rows below each construct's column
				definition = self.constructs[construct]["definition"]
				definition_references = self.constructs[construct]["definition_references"]
				examples = self.constructs[construct]["examples"]
				prompt_name = self.constructs[construct]["prompt_name"]
				metadata_df = pd.DataFrame(
					[prompt_name, definition, definition_references, examples], columns=[construct]
				)
				metadata_df[f"{construct}_include"] = [""] * len(metadata_df)
				metadata_df[f"{construct}_add"] = [""] * len(metadata_df)
				metadata_df_all.append(metadata_df)

			metadata_df_all = pd.concat(metadata_df_all, axis=1)
			lexicon_df = pd.concat([metadata_df_all, lexicon_df], axis=0, ignore_index=True)

			metadata_indeces = ["Prompt name", "Definition", "Reference", "Examples"]
			new_index = metadata_indeces + [
				n - len(metadata_indeces) for n in lexicon_df.index[len(metadata_indeces) :].tolist()
			]
			lexicon_df.index = new_index

		return lexicon_df

	def save(
		self,
		path,
		output_format=["pickle", "json", "json_metadata", "csv", "csv_annotation"],
		order=None,
		timestamp=True,
	):
		if timestamp:
			if isinstance(timestamp, str):
				path += path + f"_{timestamp}"
			elif timestamp is True:
				timestamp = generate_timestamp()
			path += f"_{timestamp}"

		if "pickle" in output_format:
			dill.dump(self, file=open(path + ".pickle", "wb"))  # save as object
		if "json" in output_format:
			save_json(self.constructs, path, with_metadata=True, order=order)
		if "json_metadata" in output_format:
			save_json(self.constructs, path, with_metadata=False, order=order)
		if "csv" in output_format:
			lexicon_df = self.to_pandas(add_annotation_columns=False, order=order)
			lexicon_df.to_csv(path + ".csv")
		if "csv_annotation" in output_format:
			lexicon_df = self.to_pandas(add_annotation_columns=True, order=order)
			lexicon_df.to_csv(path + "_annotation.csv")


def generate_timestamp(format="%y-%m-%dT%H-%M-%S"):
	ts = datetime.datetime.utcnow().strftime(format)  # so you don't overwrite, and save timestamp
	return ts


def load_lexicon(path):
	lexicon = dill.load(open(path, "rb"))
	for c in lexicon.constructs:
		tokens = lexicon.constructs[c]["tokens"]
		tokens_str = []
		for token in tokens:
			if type(token) == np.str_:
				token = token.item()
				tokens_str.append(token)
			else:
				tokens_str.append(token)
		lexicon.constructs[c]["tokens"] = tokens

	return lexicon


def warn_missing(dictionary, order, output_format=None):
	missing = [n for n in dictionary if n not in order]
	if len(missing) > 0:
		warnings.warn(
			f"These constructs were NOT SAVED in {output_format} because were not in order argument: {missing}. They"
			" are saved in the lexicon pickle file"
		)
	return


def save_json(dictionary, path, with_metadata=True, order=None):
	if order:
		warn_missing(dictionary, order, output_format="json")
		dictionary = {k: dictionary[k] for k in order}

	if with_metadata:
		with open(path + "_metadata.json", "w") as fp:
			json.dump(dictionary, fp, indent=4)
	else:
		dictionary_wo_metadata = {}
		for construct in dictionary:
			dictionary_wo_metadata[construct] = dictionary[construct].copy()
			dictionary_wo_metadata[construct]["tokens_metadata"] = "see metadata file for tokens and sources"
			dictionary_wo_metadata[construct][
				"remove"
			] = "see metadata file for tokens that were removed through human annotation/coding"
		# for source in dictionary_wo_metadata[construct]['tokens_metadata'].keys():
		# 	del dictionary_wo_metadata[construct]['tokens_metadata'][source]['tokens']

		with open(path + ".json", "w") as fp:
			json.dump(dictionary_wo_metadata, fp, indent=4)
	return


# Look for code where I obtain window for tokens in a dataset (in word scores or create_lexicon ipynb)


# Extract
# ========================================================================

#
# import pandas as pd
# import re
# from collections import Counter
# from .utils.count_words import word_count
# # from text.utils.count_words import word_count
# import numpy as np


# return_matches = True
# normalize = False
# features, matches = lexicons.extract(docs,lexicons_d, normalize = normalize, return_matches=return_matches)

# Check for false positives
# =======================================
import string


def count_lexicons_in_doc(
	doc, tokens=[], return_zero=[], return_matches=False, exact_match_n=4, exact_match_tokens=[], starts_with=None
):
	# TODO, perhaps return match and context (3 words before and after)
	"""

	Args:
									doc:
									tokens: lexicon tokens
									return_zero:
									normalize:
									return_matches:

	Returns:

	"""

	# remove punctuation except apostrophes because we need to search for things like "don't want to live"

	doc = doc.lower().replace("-", " ")
	table = str.maketrans("", "", string.punctuation.replace("'", ""))  # Apostrophe is preserved
	text = doc.translate(table)

	# text = re.sub("[^\\w\\d'\\s]+", "", doc.lower())

	counter = 0
	matched_tokens = []
	for token in tokens:
		token = token.lower()
		# if isinstance(starts_with, list):
		# 	TODO: include not only if exact match, but also if starts with
		if len(token) <= exact_match_n or token in exact_match_tokens:
			matches = len([n for n in text.split(" ") if n == token])
		else:
			matches = text.count(token)
		counter += matches
		if return_matches and matches > 0:
			# print(matches, token)
			matched_tokens.append(token)

	if return_matches:
		return counter, matched_tokens
	else:
		return counter


def remove_tokens_containing_token(tokens, except_exact_match=[]):
	"""

	tokens = remove_tokens_containing_token(['mourn', 'mourning', 'beat', 'beating'],  except_exact_match = ['beat'])
	Should keep beat and beating because beat is in except_exact_match, so it won't be redundant if you search for both.
	:param tokens:
	:param except_exact_match:
	:return:
	"""
	if len(except_exact_match) > 0:
		tokens_substrings = [n for n in tokens if n not in except_exact_match]
		partial_matching_strings, matched_substrings = find_partial_matching_strings(tokens, tokens_substrings)
		# partial matching strings contain other strings, so they are redundant
	else:
		partial_matching_strings, matched_substrings = find_partial_matching_strings(tokens, tokens)
		# partial matching strings contain other strings, so they are redundant
	tokens = [n for n in tokens if n not in partial_matching_strings]  # remove tokens containing tokens
	return tokens


# TODO: do token lemmatization outside of extract in case they want to do extract multiple times on different docs using the same lexicon

# TODO: maybe implement this where I use regex to do the counting? https://github.com/kristopherkyle/SEANCE/blob/89213d9ab7e397a64db1fde91ef7f44494a19e35/SEANCE_1_2_0.py#L403
# TODO: negation
def extract(
	docs,
	lexicon_dict,
	normalize=True,
	return_zero=[],
	return_matches=True,
	add_word_count=True,
	add_lemmatized_lexicon=True,
	lemmatize_docs=False,
	exact_match_n=4,
	exact_match_tokens=[],
):
	# TODO: return zero is for entire docs, shouldnt it be for tokens?
	"""

	Args:
									docs:
									lexicon_dict:
									normalize:
																	divide by zero
									return_zero:

	Returns:

	"""
	# process all posts
	# docs is list of list
	# lexicon_dict is dictionary {'construct':[token1, token2, ...], 'construct2':[]}
	docs = [doc.replace("\n", " ").replace("  ", " ").replace("“", "").replace("”", "") for doc in docs]
	if lemmatize_docs:
		print("lemmatizing docs...")
		docs = lemmatizer.spacy_lemmatizer(docs, language="en")  # custom function
		docs = [" ".join(n) for n in docs]

	print("extracting... ")
	docs2 = docs.copy()
	docs = []
	for doc in docs2:
		if "ness" in doc:
			# TODO: should be optional
			# No words really start with ness
			# if ' ness' in doc or doc.lower().startswith('ness'):
			# 	# eg., nec
			# 	continue
			# else:
			ness_tokens = [
				word.strip(" ,.!?*%)]|>#") for word in doc.split(" ") if word.strip(" ,.!?*%)]|>#").endswith("ness")
			]  # ['sadness,', 'loneliness,']

			tokens_adj = [word.replace("iness", "y").replace("ness", "") for word in ness_tokens]  # ['sad,', 'lonely']
			for token_ness, token_adj in zip(ness_tokens, tokens_adj):
				doc = doc.replace(token_ness, f"{token_ness} [{token_adj}]")
		docs.append(doc)

	# feature_names = list(lexicon_dict.keys())
	# full_column_names = list(df_subreddit.columns) + feature_names
	# subreddit_features = pd.DataFrame(columns=full_column_names)

	# word_counts = reddit_data.n_words.values
	# all words in subgroup

	feature_vectors = {}
	matches = {}
	matches_per_construct = {}
	matches_per_doc = {}
	for i in range(len(docs)):
		matches_per_doc[i] = {}

	for construct in tqdm(list(lexicon_dict.keys()), position=0):
		# if lemmatize_lexicon:

		# 	lexicon_tokens = lemmatizer.spacy_lemmatizer(lexicon_tokens, language='en') # custom function
		# 	lexicon_tokens = [' '.join(n) for n in lexicon_tokens]

		if add_lemmatized_lexicon:
			# replace lexicon_tokens with lemmatized tokens

			lexicon_tokens = lexicon_dict.get(construct)["tokens_lemmatized"]
			if lexicon_tokens == []:
				# Lemmatize
				warnings.warn(
					"Lemmatizing the tokens. We recommend you lemmatize before extracting so you can save time if you"
					" want to repeat extraction on different documents."
				)
				lexicon_tokens = lexicon_dict.get(construct)["tokens"]
				# If you add lemmatized and nonlemmatized you'll get double count in many cases ("plans" in doc will be matched by "plan" and "plans" in lexicon)

				lexicon_tokens_lemmatized = lemmatizer.spacy_lemmatizer(
					lexicon_tokens, language="en"
				)  # custom function
				lexicon_tokens_lemmatized = [" ".join(n) for n in lexicon_tokens_lemmatized]
				lexicon_tokens += lexicon_tokens_lemmatized
				lexicon_tokens = list(np.unique(lexicon_tokens))  # unique set

				# lexicon_dict[construct]['tokens_lemmatized']=lexicon_tokens
			# If you add lemmatized and nonlemmatized you'll get double count in many cases ("plans" in doc will be matched by "plan" and "plans" in lexicon)
			# 	lexicon_tokens_lemmatized = lemmatizer.spacy_lemmatizer(lexicon_tokens, language='en') # custom function
			# 	lexicon_tokens_lemmatized = [' '.join(n) for n in lexicon_tokens_lemmatized]
			# 	lexicon_tokens += lexicon_tokens_lemmatized
			# 	lexicon_tokens = list(np.unique(lexicon_tokens)) # unique set

		else:
			lexicon_tokens = lexicon_dict.get(construct)["tokens"]

			"""
			lemmatizer.spacy_lemmatizer(['distressed'])
			'distressed' > "distress", only "distress" is kept, to avoid counting twice.
			"drained" > "drain", only "drain" is kept unless its in the except_exact_match list"
			"die" and "died" will be kep because they are in the exact match list, because <= exact_match_n
			"catastrophizing" > "catastrophize", both are kept
			'forced to exist' > 'force to exist'
			"I'm a drag"> "I am a drag", both will be kept, because one is not a substring of the other
			"grabbed me"> "grab me", both will be kept, because one is not a substring of the other
			"""
		# remove tokens that contain tokens to avoid counting twice
		# except for exact_match_n and exact matches.
		except_exact_match = list(
			np.unique(exact_match_tokens + [n for n in lexicon_tokens if len(n) <= exact_match_n])
		)  # TODO: maybe "died">"die"
		lexicon_tokens = remove_tokens_containing_token(lexicon_tokens, except_exact_match=except_exact_match)

		if return_matches:
			counts_and_matched_tokens = [
				count_lexicons_in_doc(
					doc,
					tokens=lexicon_tokens,
					return_zero=return_zero,
					return_matches=return_matches,
					exact_match_n=exact_match_n,
					exact_match_tokens=exact_match_tokens,
				)
				for doc in docs
			]
			matches_per_construct[construct] = counts_and_matched_tokens
			# for a single construct
			for i, doc_i in enumerate(counts_and_matched_tokens):
				# each document for that construct
				matches_per_doc[i][construct] = doc_i

			counts = [n[0] for n in counts_and_matched_tokens]
			matched_tokens = [n[1] for n in counts_and_matched_tokens if n[1] != []]
			matches[construct] = matched_tokens

		else:
			counts = [
				count_lexicons_in_doc(
					doc,
					tokens=lexicon_tokens,
					return_zero=return_zero,
					return_matches=return_matches,
					exact_match_n=exact_match_n,
					exact_match_tokens=exact_match_tokens,
				)
				for doc in docs
			]
		# one_construct = one_construct/word_counts #normalize

		feature_vectors[construct] = counts

	# # feature_vector = extract_NLP_features(post, features) #removed feature_names from output
	# if len(feature_vector) != 0:
	#     raw_series = list(df_subreddit.iloc[pi])
	#     subreddit_features = subreddit_features.append(pd.Series(raw_series + feature_vector, index=full_column_names), ignore_index=True)

	# feature_vectors0   = pd.DataFrame(docs, columns = ['docs'])
	# feature_vectors = pd.concat([feature_vectors0,pd.DataFrame(feature_vectors)],axis=1)
	feature_vectors = pd.DataFrame(feature_vectors)

	#     feature_vectors   = pd.DataFrame(docs)
	#     feature_vectors['docs']=docs

	if normalize:
		wc = word_count(docs, return_zero=return_zero)
		wc = np.array(wc)
		feature_vectors_normalized = np.divide(feature_vectors.values.T, wc).T
		feature_vectors = pd.DataFrame(
			feature_vectors_normalized, index=feature_vectors.index, columns=feature_vectors.columns
		)

	if add_word_count and normalize:
		feature_vectors["word_count"] = wc
	elif add_word_count and not normalize:
		wc = word_count(docs, return_zero=return_zero)
		feature_vectors["word_count"] = wc

	# feature_vectors = feature_vectors/wc

	if return_matches:
		# all lexicons
		matches_counter_d = {}
		for lexicon_name_i in list(lexicon_dict.keys()):
			if matches.get(lexicon_name_i):
				x = Counter([n for i in matches.get(lexicon_name_i) for n in i])
				matches_counter_d[lexicon_name_i] = {
					k: v for k, v in sorted(x.items(), key=lambda item: item[1], reverse=True)
				}
		# Counter([n for i in matches.get(lexicon_name_i) for n in i]) for lexicon_name_i in lexicon_dict.keys()]

		return feature_vectors, matches_counter_d, matches_per_doc, matches_per_construct
	else:
		return feature_vectors


import re


def find_match(s, pat):
	pat = r"(\w*%s\w*)" % pat  # Not thrilled about this line
	match = re.findall(pat, s)
	return match


def startswith_str(doc, pat):
	tokens = doc.split(" ")
	match = [n for n in tokens if n.startswith(pat)]
	return match


def match(docs, token):
	matches = [find_match(doc.lower(), token) for doc in docs]
	# TODO: don't lower acronyms
	# matches = [startswith_str(doc.lower(), token) for doc in docs]
	matches = list(np.unique([n for i in matches for n in i]))
	return matches


def lemmatize_tokens(lexicon_object):
	# TODO: do not lemmatize "I", hit me > hit I, cut my > cut I, same with her and him> block he, block her. UNLESS you lemmatize the doc as well.
	for c in tqdm(lexicon_object.constructs.keys()):
		srl_tokens = lexicon_object.constructs[c]["tokens"].copy()
		# If you add lemmatized and nonlemmatized you'll get double count in many cases ("plans" in doc will be matched by "plan" and "plans" in srl)
		srl_tokens_lemmatized = lemmatizer.spacy_lemmatizer(srl_tokens, language="en")  # custom function
		srl_tokens_lemmatized = [" ".join(n) for n in srl_tokens_lemmatized]
		srl_tokens += srl_tokens_lemmatized
		if lexicon_object.constructs[c]["remove"] is None:
			lexicon_object.constructs[c]["remove"] = []
		srl_tokens = [
			n.replace(" - ", "-").strip() for n in srl_tokens if n not in lexicon_object.constructs[c]["remove"]
		]
		srl_tokens = [
			n.replace("-", " ").strip() for n in srl_tokens if n not in lexicon_object.constructs[c]["remove"]
		]
		srl_tokens = list(np.unique(srl_tokens))  # unique set

		# TODO: add back in override_remove

		lexicon_object.constructs[c]["tokens_lemmatized"] = srl_tokens
	return lexicon_object


# Todo
"""
add section: duplicates:
add section: manually added: (if section=='tokens' and value != 'None', add count to manually_added
Use code from construct_text_similarity.ipynb to save clean version of SEANCE
clean_seance_names.csv
inquirerbasic.txt
add num_retries=2 to corresponding functions

"""
