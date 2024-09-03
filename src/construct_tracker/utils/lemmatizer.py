"""Lemmatize documents."""

import subprocess
from typing import List, Union

import numpy as np
import spacy


def spacy_lemmatizer(docs: List[Union[str, np.str_]], language: str = "en") -> List[List[str]]:
    """
    Lemmatizes a list of documents using spaCy.

    Args:
        docs (List[Union[str, np.str_]]): A list of documents to lemmatize. Each document can be a string or a NumPy string object.
        language (str, optional): The language model to use for lemmatization. Defaults to "en".

    Returns:
        List[List[str]]: A list of lists, where each inner list contains the lemmatized tokens of the corresponding document.

    Example:
        docs = ['alone', "I've been worried but hopeful", "I've been feeling all alone but hopeful and I'll do therapy. Gotta take it step by step.", "I've been feeling all alone but hopeful and I'll do therapy. Gotta take it step by step."]
        docs = stop_words.remove(docs)
        docs = spacy_lemmatizer(docs)
        print(docs)
        # Output: [['alone'], ['worried', 'hopeful'], ['feel', 'alone', 'hopeful', 'ill', 'therapy', 'got', 'to', 'take', 'step', 'step']]
    """
    if language == "en":
        try:
            nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except Exception as e:
            print(f"Error: {e}. Downloading spaCy model: en_core_web_sm")
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            print("Finished downloading")
            nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    docs_lemmatized = []

    for doc in docs:
        if isinstance(doc, np.str_):
            doc = doc.item()
        spacy_doc = nlp(doc)

        doc_lemmatized = [token.lemma_ for token in spacy_doc]
        docs_lemmatized.append(doc_lemmatized)

    return docs_lemmatized
