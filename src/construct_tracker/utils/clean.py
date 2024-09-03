"""Functions to clean strings."""

import re
import string


def remove_multiple_spaces(doc: str) -> str:
    """
    Removes consecutive spaces in a document and replaces them with a single space.

    Args:
        doc (str): The document from which to remove multiple spaces.

    Returns:
        str: The document with consecutive spaces replaced by a single space.

    Example:
        >>> remove_multiple_spaces("This  is   a   test.")
        'This is a test.'
    """
    return re.sub(" +", " ", doc)


def remove_punctuation(doc: str) -> str:
    """
    Removes all punctuation from a given document.

    Args:
        doc (str): The document from which to remove punctuation.

    Returns:
        str: The document with all punctuation removed.

    Example:
        >>> remove_punctuation("Hello, world!")
        'Hello world'
    """
    return doc.translate(str.maketrans("", "", string.punctuation))


def remove_extra_white_space(doc: str) -> str:
    """
    Removes extra spaces around punctuation marks in a document, ensuring proper spacing.

    Args:
        doc (str): The document from which to remove extra spaces.

    Returns:
        str: The document with extra spaces removed around punctuation marks.

    Example:
        >>> remove_extra_white_space("Hello , world !")
        'Hello, world!'
    """
    punctuations_closing = [".", ",", "!", "?", "]", ")"]
    punctuations_opening = ["(", "[", "$"]

    for punctuation in punctuations_closing:
        doc = doc.replace(" " + punctuation, punctuation)
    for punctuation in punctuations_opening:
        doc = doc.replace(punctuation + " ", punctuation)
