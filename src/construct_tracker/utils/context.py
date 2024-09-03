"""Get window of string."""

from typing import List, Optional, Tuple


def get_context(doc: str, token: str, n_words_pre: int = 10, n_words_post: int = 10) -> str:
    """
    Extracts a window of text around a given token within a document.

    Args:
        doc (str): The document from which to extract the context.
        token (str): The token around which to extract the context.
        n_words_pre (int, optional): The number of words to include before the token. Defaults to 10.
        n_words_post (int, optional): The number of words to include after the token. Defaults to 10.

    Returns:
        str: A string containing the specified number of words before and after the token.

    Example:
        >>> get_context("This is a test document for extracting context.", "test", 2, 3)
        'is a test document for extracting'
    """
    doc_pre_token = " ".join(doc.split(token)[0].split(" ")[-n_words_pre:])
    doc_post_token = " ".join(doc.split(token)[1].split(" ")[:n_words_post])
    doc_windowed = doc_pre_token + token + doc_post_token
    return doc_windowed


def get_docs_matching_token(
    docs: List[str], token: str, window: Optional[Tuple[int, int]] = (10, 10), exact_match_n: int = 4
) -> List[str]:
    """
    Filters a list of documents to those that contain a specific token, with options
    for exact matching and extracting context windows around the token.

    Args:
        docs (List[str]): A list of documents to search.
        token (str): The token to search for within the documents.
        window (Optional[Tuple[int, int]], optional): A tuple specifying the number of words before and after
                                                     the token to include in the context. Defaults to (10, 10).
        exact_match_n (int, optional): If the token length is less than or equal to this number,
                                       only exact matches (whole words) will be considered. Defaults to 4.

    Returns:
        List[str]: A list of documents or document snippets that match the token criteria.

    Example:
        >>> get_docs_matching_token(['get paranoid and I think this is also a'], 'thin', window=(10,10), exact_match_n=4)
        ['get paranoid and I think this is also a']
    """
    docs_matching_token = [n for n in docs if token in n]

    if len(token) <= exact_match_n:
        # Exact match
        docs_matching_token2 = docs_matching_token.copy()
        docs_matching_token = []
        for doc in docs_matching_token2:
            words = doc.split(" ")
            if token in words:
                docs_matching_token.append(doc)

    if window:
        docs_matching_token_windowed = []
        for doc in docs_matching_token:
            doc_windowed = get_context(doc, token, n_words_pre=window[0], n_words_post=window[1])
            docs_matching_token_windowed.append(doc_windowed)
        return docs_matching_token_windowed
    else:
        return docs_matching_token
