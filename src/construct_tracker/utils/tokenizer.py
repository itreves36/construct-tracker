"""

Source: https://stackoverflow.com/questions/65227103/clause-extraction-long-sentence-segmentation-in-python

Alternatives:
- second response: https://stackoverflow.com/questions/39320015/how-to-split-an-nlp-parse-tree-to-clauses-independent-and-subordinate
- TODO: also consider subordinate clauses while, if, becuase, instead https://stackoverflow.com/questions/68616708/how-to-split-sentence-into-clauses-in-python

"""



import deplacy
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
    Tokenizes a list of documents using the SpaCy library.

    Args:
        docs (List[str]): A list of documents to be tokenized.
        language (str, optional): The language of the documents. Defaults to "en".
        model (str, optional): The SpaCy model to use for tokenization. Defaults to "en_core_web_sm".
        method (str, optional): The tokenization method to use. Can be "unigram", "clause", or "sentence". Defaults to "clause".
        lowercase (bool, optional): Whether to convert tokens to lowercase. Defaults to False.
        display_tree (bool, optional): Whether to display the parsed tree. Defaults to False.
        remove_punct (bool, optional): Whether to remove punctuation from the tokens. Defaults to True.
        clause_remove_conj (bool, optional): Whether to remove coordinating conjunctions from clause tokens. Defaults to True.

    Returns:
        List[List[str]]: A list of tokenized documents. The structure of the list depends on the tokenization method used.
            - If method is "unigram", returns a list of lists, where each inner list contains the tokens of a document.
            - If method is "clause", returns a list of lists, where each inner list contains the clause tokens of a document.
            - If method is "sentence", returns a list of lists, where each inner list contains the sentence tokens of a document.
    """

    # TODO: split if you find ";"
    # TODO: make into list comprehensions for faster processing
    if method == "unigram":
        my_module = importlib.import_module("spacy.lang." + language)
        if language == "en":
            nlp = my_module.English()
        tokens_for_all_docs = []
        for doc in docs:
            doc = nlp(doc)
            tokens = [token.text.lower() for token in doc] if lowercase else [token.text for token in doc]
            tokens_for_all_docs.append(tokens)
        return tokens_for_all_docs

    elif method == "clause":
        nlp = spacy.load(model)
        chunks_for_all_docs = []
        for doc in nlp.pipe(docs):
            if display_tree:
                import deplacy
                print(doc)
                print(deplacy.render(doc))

            seen = set()
            chunks = []
            for sent in doc.sents:
                heads = [cc for cc in sent.root.children if cc.dep_ == "conj"]

                for head in heads:
                    words = list(head.subtree)
                    if remove_punct:
                        words = [n for n in words if not n.is_punct]
                    for word in words:
                        seen.add(word)
                    if clause_remove_conj:
                        chunk = []
                        for i, word in enumerate(words):
                            if not (word.tag_ == "CC" and i == len(words) - 1):
                                chunk.append(word.text)
                        chunk = " ".join(chunk)
                    else:
                        chunk = " ".join([ww.text for ww in words])
                    chunks.append((head.i, chunk))

                unseen = [ww for ww in sent if ww not in seen]
                if remove_punct:
                    unseen = [n for n in unseen if not n.is_punct]
                if clause_remove_conj:
                    chunk = []
                    for i, word in enumerate(unseen):
                        if not (word.tag_ == "CC" and i == len(unseen) - 1):
                            chunk.append(word.text)
                    chunk = " ".join(chunk)
                else:
                    chunk = " ".join([ww.text for ww in unseen])
                chunks.append((sent.root.i, chunk))

            chunks = sorted(chunks, key=lambda x: x[0])
            chunks = [n[1] for n in chunks]
            chunks_for_all_docs.append(chunks)

        docs_clauses_clean = [
            [clause.replace(' ,', ', ').replace(" 's", "'s").replace('  ', ' ').strip(', ') for clause in doc]
            for doc in chunks_for_all_docs
        ]

        return docs_clauses_clean

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



   