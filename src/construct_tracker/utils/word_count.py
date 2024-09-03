"""Count words in documents."""

import string
from typing import List, Union


def word_count(docs: List[str], return_zero: List[str] = None) -> Union[int, List[int]]:
    """
    Counts the number of words in each document, considering only alphabetic words.

    Args:
        docs (List[str]): A list of documents (strings) to count words in.
        return_zero (List[str], optional): A list of documents that should return a word count of zero. Defaults to None.

    Returns:
        Union[int, List[int]]: A list of word counts for each document, or a single integer if only one document is provided.

    Example:
        >>> word_count(["Hello, world!", "R.I.P.", "", "This is a test."])
        [2, 1, 0, 4]
    """
    if return_zero is None:
        return_zero = []

    word_counts = []
    for doc_i in docs:
        doc_i = str(doc_i)
        if doc_i in return_zero or len(doc_i) == 0:
            word_counts.append(0)
        else:
            # Check if all tokens are non-alphabetic (e.g., acronyms like 'R.I.P.')
            if sum([i.strip(string.punctuation).isalpha() for i in doc_i.split()]) == 0:
                word_counts.append(1)
            else:
                wc_i = sum([i.strip(string.punctuation).isalpha() for i in doc_i.split()])
                word_counts.append(wc_i)

    if len(word_counts) == 1:
        return word_counts[0]
    else:
        return word_counts
