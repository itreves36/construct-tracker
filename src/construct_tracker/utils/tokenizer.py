"""

Source: https://stackoverflow.com/questions/65227103/clause-extraction-long-sentence-segmentation-in-python

Alternatives:
- second response: https://stackoverflow.com/questions/39320015/how-to-split-an-nlp-parse-tree-to-clauses-independent-and-subordinate
- TODO: also consider subordinate clauses while, if, becuase, instead https://stackoverflow.com/questions/68616708/how-to-split-sentence-into-clauses-in-python

"""

import importlib
import spacy



def spacy_tokenizer(
    docs,
    language="en",
    model="en_core_web_sm",
    method="clause",
    lowercase=False,
    display_tree=False,
    remove_punct=True,
    clause_remove_conj=True,
):
    """
    This function tokenizes documents using spacy.

    Args:
        docs (list): List of documents to be tokenized.
        language (str): Language of the documents. Defaults to 'en'.
        model (str): Model to be used for tokenization. Defaults to 'en_core_web_sm'.
        method (str): Method of tokenization. Possible values: {'clause', 'unigram', 'sentence'}.
        lowercase (bool): Whether to convert tokens to lowercase. Defaults to False.
        display_tree (bool): Whether to display the parse tree of the documents. Defaults to False.
        remove_punct (bool): Whether to remove punctuation from tokens. Defaults to True.
        clause_remove_conj (bool): Whether to remove coordinating conjunctions from clauses. Defaults to True.

    Returns:
        list: List of tokenized documents.
    """
    
    # TODO: split if you find ";"
    # TODO: make into list comprehensions for faster processing
    if method == "unigram":
        # doc = 'I am a boy'
        my_module = importlib.import_module("spacy.lang." + language)  # from spacy.lang.en import English
        if language == "en":
            nlp = my_module.English()
        tokens_for_all_docs = []
        for doc in docs:
            doc = nlp(doc)
            tokens = [token.text.lower() for token in doc] if lowercase else [token.text.lower() for token in doc]
            tokens_for_all_docs.append(tokens)
        return tokens_for_all_docs
    elif method == "clause":
        nlp = spacy.load(model)
        chunks_for_all_docs = []
        for doc in nlp.pipe(docs):
            # doc = en(text)
            if display_tree:
                import deplacy
                print(doc)
                print(deplacy.render(doc))

            seen = set()  # keep track of covered words
            chunks = []
            for sent in doc.sents:
                # Extract independent clauses
                heads = [cc for cc in sent.root.children if cc.dep_ == "conj"]

                for head in heads:
                    words = list(head.subtree)
                    if remove_punct:
                        words = [n for n in words if not n.is_punct]
                    for word in words:
                        seen.add(word)
                    if clause_remove_conj:
                        chunk = [ww for ww in words if ww.tag_ != "CC" or ww != words[-1]]
                    else:
                        chunk = [ww.text for ww in words]
                    chunks.append(" ".join(chunk))

                # Extract remaining words
                unseen = [ww for ww in sent if ww not in seen]
                if remove_punct:
                    unseen = [n for n in unseen if not n.is_punct]
                if clause_remove_conj:
                    chunk = [ww for ww in unseen if ww.tag_ != "CC" or ww != unseen[-1]]
                else:
                    chunk = [ww.text for ww in unseen]
                chunks.append(" ".join(chunk))

                # Sort chunks by the order of appearance
                chunks = sorted(chunks, key=lambda x: sent.start_char + x.start)
            chunks_for_all_docs.append(chunks)
        docs_tokenized = []
        for doc in chunks_for_all_docs:
            # Clean up the tokenization
            doc_clean = [clause.replace(' ,', ', ').replace(" 's", "'s").replace('  ', ' ').strip(', ') for clause in doc]
            docs_tokenized.append(doc_clean)
        return docs_tokenized
    elif method == 'sentence':
        nlp = spacy.load(model)
        docs_tokenized = [[sent.text for sent in nlp(string).sents] for string in docs]
        return docs_tokenized


"""

docs_tokenized = spacy_tokenizer(docs, language = 'en', model='en_core_web_sm',
					method = 'clause',lowercase=False, display_tree = True,
					remove_punct=True, clause_remove_conj = True)
print(docs_tokenized)


docs_long = [
    "I've been feeling all alone and I feel like a burden to my family. I'll do therapy, but I'm pretty hopeless.", 
    'I am very sad but hopeful and I will start therapy', 
    'I am very sad, but hopeful and I will start therapy',
    "I've been feeling all alone but hopeful and I'll do therapy. Gotta take it step by step."
]


docs_long_clauses = spacy_tokenizer(docs_long, 
										language = 'en', model='en_core_web_sm', 
										method = 'clause', # clause tokenization
										lowercase=False, 
										display_tree = True, 
										remove_punct=False, 
										clause_remove_conj = True)

docs_long_clauses
"""