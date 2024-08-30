"""Tokenize strings.

Source: https://stackoverflow.com/questions/65227103/clause-extraction-long-sentence-segmentation-in-python

Alternatives:
- second response: https://stackoverflow.com/questions/39320015/how-to-split-an-nlp-parse-tree-to-clauses-independent-and-subordinate
- TODO: also consider subordinate clauses while, if, because, instead https://stackoverflow.com/questions/68616708/how-to-split-sentence-into-clauses-in-python

Author: Daniel M. Low
License: Apache 2.0.
"""

import subprocess
import sys
import re
import deplacy
import spacy
import tqdm


def spacy_tokenizer(
    docs,
    nlp=None,
    method="clause",
    lowercase=False,
    display_tree=False,
    remove_punct=True,
    clause_remove_conj=True,
):
    if nlp is None:
        try:
            model = "en_core_web_sm"
            nlp = spacy.load(model)
        except:
            print(f"Model {model} not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model])
            model = "en_core_web_sm"
            nlp = spacy.load(model)

    if method == "word":
        tokens_for_all_docs = [[token.text.lower() if lowercase else token.text for token in nlp(doc)] for doc in docs]
        return tokens_for_all_docs

    elif method == "clause":
        chunks_for_all_docs = []
        for doc in tqdm.tqdm(nlp.pipe(docs, batch_size=2048), position=0):
            if display_tree:
                print(doc)
                print(deplacy.render(doc))
                # Note: Only print the tree if necessary, as it slows down the process.

            seen = set()  # keep track of covered words
            chunks = []
            for sent in doc.sents:
                heads = [cc for cc in sent.root.children if cc.dep_ == "conj"]

                for head in heads:
                    words = [n for n in head.subtree if not (remove_punct and n.is_punct)]
                    seen.update(words)

                    if clause_remove_conj:
                        words = [
                            word for i, word in enumerate(words) if not (word.tag_ == "CC" and i == len(words) - 1)
                        ]
                    chunks.append((head.i, " ".join([ww.text for ww in words])))

                unseen = [ww for ww in sent if ww not in seen and not (remove_punct and ww.is_punct)]
                if clause_remove_conj:
                    unseen = [word for i, word in enumerate(unseen) if not (word.tag_ == "CC" and i == len(unseen) - 1)]
                chunks.append((sent.root.i, " ".join([ww.text for ww in unseen])))

            chunks_for_all_docs.append([n[1] for n in sorted(chunks, key=lambda x: x[0])])

        docs_clauses_clean = [
            [
                clause.replace(" ,", ",")
                .replace(" .", ".")
                .replace(" !", "!")
                .replace(" ?", "?")
                .replace(" '", "'")
                .replace("  ", " ")
                .strip(", ")
                for clause in doc
            ]
            for doc in chunks_for_all_docs
        ]

        return docs_clauses_clean

    elif method == "sentence":
        docs_tokenized = [[sent.text for sent in nlp(string).sents] for string in docs]
        return docs_tokenized


"""




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


# def nltk_lemmatize(text):
# 	from nltk import word_tokenize
# 	from nltk.stem import WordNetLemmatizer
# 	lemmatizer = WordNetLemmatizer()
#     return [lemmatizer.lemmatize(w) for w in word_tokenize(text)]

# from sklearn.feature_extraction.text import TfidfVectorizer

# tfidf_vectorizer = TfidfVectorizer(tokenizer=nltk_lemmatize, stop_words='english')



def custom_tokenizer(string):
    from nltk.tokenize import RegexpTokenizer
    tokenizer = RegexpTokenizer(r"\w+")
    words = tokenizer.tokenize(string)
    return words


def tokenizer_remove_punctuation(text):
    return re.split("\\s+", text)
