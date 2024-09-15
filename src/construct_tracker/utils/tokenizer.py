"""
Tokenize strings.

Source: https://stackoverflow.com/questions/65227103/clause-extraction-long-sentence-segmentation-in-python

Alternatives:
- Second response: https://stackoverflow.com/questions/39320015/how-to-split-an-nlp-parse-tree-to-clauses-independent-and-subordinate
- TODO: Also consider subordinate clauses: while, if, because, instead: https://stackoverflow.com/questions/68616708/how-to-split-sentence-into-clauses-in-python

Author: Daniel M. Low
License: Apache 2.0.
"""

import re
import subprocess
import sys
from typing import List, Optional, Union

import spacy
import tqdm


def spacy_tokenizer(
    docs: List[str],
    nlp: Optional[spacy.language.Language] = None,
    method: str = "clause",
    lowercase: bool = False,
    # display_tree: bool = False,
    remove_punct: bool = True,
    clause_remove_conj: bool = True,
) -> Union[List[List[str]], List[str]]:
    """
    Tokenizes documents using spaCy with options for word, sentence, or clause tokenization.

    Args:
            docs (List[str]): A list of documents to tokenize.
            nlp (Optional[spacy.language.Language], optional): Preloaded spaCy model. If None, the model will be loaded. Defaults to None.
            method (str, optional): The method of tokenization ('word', 'sentence', 'clause'). Defaults to 'clause'.
            lowercase (bool, optional): If True, tokens are converted to lowercase. Defaults to False.
            display_tree (bool, optional): If True, displays the dependency tree. Defaults to False.
            remove_punct (bool, optional): If True, removes punctuation from tokens. Defaults to True.
            clause_remove_conj (bool, optional): If True, removes conjunctions at the end of clauses. Defaults to True.

    Returns:
            Union[List[List[str]], List[str]]: Tokenized documents as a list of lists of tokens.

    Example:
            >>> docs = ["I am happy but tired. I will rest."]
            >>> spacy_tokenizer(docs, method="clause")
            [['I am happy', 'tired.', 'I will rest.']]
            # Try these:
            docs_long = [
                                            "I've been feeling all alone and I feel like a burden to my family. I'll do therapy, but I'm pretty hopeless.",
                                            'I am very sad but hopeful and I will start therapy',
                                            'I am very sad, but hopeful and I will start therapy',
                                            "I've been feeling all alone but hopeful and I'll do therapy. Gotta take it step by step."
                            ]
    """
    if method not in ["word", "clause", "sentence"]:
        print("Warning: method not found. Available methods are: 'word', 'clause', 'sentence'")
        return None
    if nlp is None:
        model = "en_core_web_sm"
        try:
            nlp = spacy.load(model)
        except OSError:
            print(f"Model {model} not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model])
            nlp = spacy.load(model)

    if method == "word":
        return [[token.text.lower() if lowercase else token.text for token in nlp(doc)] for doc in docs]

    elif method == "clause":
        chunks_for_all_docs = []
        for doc in tqdm.tqdm(nlp.pipe(docs, batch_size=2048), position=0):
            # if display_tree:
            #     import deplacy
            #     print(doc)
            #     print(deplacy.render(doc))

            seen = set()
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
        return [[sent.text for sent in nlp(string).sents] for string in docs]


def custom_tokenizer(string: str) -> List[str]:
    """
    Tokenizes a string into words using a regular expression.

    Args:
            string (str): The input string to tokenize.

    Returns:
            List[str]: A list of words.

    Example:
            >>> custom_tokenizer("Hello, world!")
            ['Hello', 'world']
    """
    from nltk.tokenize import RegexpTokenizer

    tokenizer = RegexpTokenizer(r"\w+")
    words = tokenizer.tokenize(string)
    return words


def tokenizer_remove_punctuation(text: str) -> List[str]:
    """
    Tokenizes a string and removes punctuation by splitting on whitespace.

    Args:
            text (str): The input string to tokenize.

    Returns:
            List[str]: A list of tokens.

    Example:
            >>> tokenizer_remove_punctuation("Hello, world!")
            ['Hello,', 'world!']
    """
    return re.split(r"\s+", text)
