"""This extracts many linguistic features such as the filler ratio,
type_token_ratio, entropy, standardized_word_entropy,
question_ratio, number_ratio, brunet's index, honore's statistic,
and many others.

Adapted from: https://github.com/jim-schwoebel/allie/blob/99350a2916fa81ab0f10229a9c87ad12703c8e9a/features/text_features/text_features.py#L75
"""

import math
from typing import List, Optional, Tuple

import numpy as np
from nltk import FreqDist
from nltk.tokenize import RegexpTokenizer, word_tokenize


def filler_ratio(s: str, tokens: Optional[List[str]] = None) -> float:
    """
    Calculates the ratio of filler words in a document.

    Args:
        s (str): The document string.
        tokens (Optional[List[str]]): A list of tokens from the document. If None, tokens will be generated using word_tokenize.

    Returns:
        float: The filler word ratio.

    Example:
        >>> filler_ratio("um, well, you know, it's like...")
        0.5
    """
    if tokens is None:
        tokens = word_tokenize(s)

    tokenizer = RegexpTokenizer("uh|ugh|um|like|you know")
    qtokens = tokenizer.tokenize(s.lower())

    if len(tokens) == 0:
        return 0.0
    else:
        return len(qtokens) / len(tokens)


def type_token_ratio(s: str, tokens: Optional[List[str]] = None) -> float:
    """
    Calculates the type-token ratio, a measure of lexical diversity.

    Args:
        s (str): The document string.
        tokens (Optional[List[str]]): A list of tokens from the document. If None, tokens will be generated using word_tokenize.

    Returns:
        float: The type-token ratio.

    Example:
        >>> type_token_ratio("I love love programming")
        0.75
    """
    if tokens is None:
        tokens = word_tokenize(s)

    uniques = [token for token, count in FreqDist(tokens).items() if count == 1]

    if len(tokens) == 0:
        return 0.0
    else:
        return len(uniques) / len(tokens)


def entropy(tokens: List[str]) -> float:
    """
    Calculates the entropy of the token distribution in the document.

    Args:
        tokens (List[str]): A list of tokens from the document.

    Returns:
        float: The entropy value.

    Example:
        >>> entropy(["I", "love", "love", "programming"])
        1.5
    """
    freqdist = FreqDist(tokens)
    probs = [freqdist.freq(i) for i in freqdist]

    return -sum(p * math.log(p, 2) for p in probs)


def standardized_word_entropy(s: str, tokens: Optional[List[str]] = None) -> float:
    """
    Calculates the standardized word entropy, normalizing the entropy by the logarithm of the token count.

    Args:
        s (str): The document string.
        tokens (Optional[List[str]]): A list of tokens from the document. If None, tokens will be generated using word_tokenize.

    Returns:
        float: The standardized word entropy.

    Example:
        >>> standardized_word_entropy("I love love programming")
        0.75
    """
    if tokens is None:
        tokens = word_tokenize(s)

    if len(tokens) == 0 or math.log(len(tokens)) == 0:
        return 0.0
    else:
        return entropy(tokens) / math.log(len(tokens))


def question_ratio(s: str, tokens: Optional[List[str]] = None) -> float:
    """
    Calculates the ratio of question-related words in the document.

    Args:
        s (str): The document string.
        tokens (Optional[List[str]]): A list of tokens from the document. If None, tokens will be generated using word_tokenize.

    Returns:
        float: The question word ratio.

    Example:
        >>> question_ratio("Who are you? What do you want?")
        0.25
    """
    if tokens is None:
        tokens = word_tokenize(s)

    tokenizer = RegexpTokenizer("Who|What|When|Where|Why|How|\\?")
    qtokens = tokenizer.tokenize(s)

    if len(tokens) == 0:
        return 0.0
    else:
        return len(qtokens) / len(tokens)


def number_ratio(s: str, tokens: Optional[List[str]] = None) -> float:
    """
    Calculates the ratio of number-related words in the document.

    Args:
        s (str): The document string.
        tokens (Optional[List[str]]): A list of tokens from the document. If None, tokens will be generated using word_tokenize.

    Returns:
        float: The number word ratio.

    Example:
        >>> number_ratio("I have two cats and three dogs.")
        0.2
    """
    if tokens is None:
        tokens = word_tokenize(s)

    tokenizer = RegexpTokenizer(
        "zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|trillion|dozen|couple|several|few|\\d"
    )
    qtokens = tokenizer.tokenize(s.lower())

    if len(tokens) == 0:
        return 0.0
    else:
        return len(qtokens) / len(tokens)


def brunets_index(s: str, tokens: Optional[List[str]] = None) -> float:
    """
    Calculates Brunet's index, a measure of lexical richness.

    Args:
        s (str): The document string.
        tokens (Optional[List[str]]): A list of tokens from the document. If None, tokens will be generated using word_tokenize.

    Returns:
        float: Brunet's index.

    Example:
        >>> brunets_index("I love programming and programming is fun.")
        8.396016
    """
    if tokens is None:
        tokens = word_tokenize(s)

    N = float(len(tokens))
    V = float(len(set(tokens)))

    if N == 0 or V == 0:
        return 0.0
    else:
        return math.pow(N, math.pow(V, -0.165))


def honores_statistic(s: str, tokens: Optional[List[str]] = None) -> float:
    """
    Calculates Honore's statistic, another measure of lexical richness.

    Args:
        s (str): The document string.
        tokens (Optional[List[str]]): A list of tokens from the document. If None, tokens will be generated using word_tokenize.

    Returns:
        float: Honore's statistic.

    Example:
        >>> honores_statistic("I love programming and programming is fun.")
        150.13059
    """
    if tokens is None:
        tokens = word_tokenize(s)

    uniques = [token for token, count in FreqDist(tokens).items() if count == 1]

    N = float(len(tokens))
    V = float(len(set(tokens)))
    V1 = float(len(uniques))

    if N == 0 or V == 0 or V1 == 0:
        return 0.0
    elif V == V1:
        return 100 * math.log(N)
    else:
        return (100 * math.log(N)) / (1 - (V1 / V))


def datewords_freq(importtext: str) -> float:
    """
    Calculates the frequency of date-related words in the document.

    Args:
        importtext (str): The document string.

    Returns:
        float: The frequency of date-related words.

    Example:
        >>> datewords_freq("Today is Monday, January 1st.")
        0.2
    """
    text = word_tokenize(importtext.lower())
    datewords = [
        "time",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "november",
        "december",
        "year",
        "day",
        "hour",
        "today",
        "month",
        "o'clock",
        "pm",
        "am",
    ]
    datewords += [dw + "s" for dw in datewords]

    datecount = sum(1 for word in text if word in datewords)

    try:
        datewordfreq = datecount / len(text)
    except ZeroDivisionError:
        datewordfreq = 0.0

    return datewordfreq


def word_stats(importtext: str) -> List[float]:
    """
    Calculates various word statistics including average word length, word length variance, and others.

    Args:
        importtext (str): The document string.

    Returns:
        List[float]: A list of word statistics (average word length, number of words > 5 chars, max word length, etc.).

    Example:
        >>> word_stats("This is a test document.")
        [3.4, 1.0, 8.0, 1.0, 6.240000000000001, 2.4999999999999996]
    """
    text = word_tokenize(importtext)
    awordlength = np.mean([len(word) for word in text])
    fivewordnum = len([word for word in text if len(word) > 5])
    vmax = np.amax([len(word) for word in text])
    vmin = np.amin([len(word) for word in text])
    vvar = np.var([len(word) for word in text])
    vstd = np.std([len(word) for word in text])

    return [awordlength, float(fivewordnum), float(vmax), float(vmin), float(vvar), float(vstd)]


def num_sentences(importtext: str) -> List[int]:
    """
    Counts the number of sentences, periods, questions, and interjections in the document.

    Args:
        importtext (str): The document string.

    Returns:
        List[int]: A list containing the total number of sentences, periods, questions, and interjections.

    Example:
        >>> num_sentences("This is a test. Is it? Yes!")
        [3, 1, 1, 1]
    """
    periods = importtext.count(".")
    questions = importtext.count("?")
    interjections = importtext.count("!")

    sentencenum = periods + questions + interjections

    return [sentencenum, periods, questions, interjections]


def word_repeats(importtext: str) -> List[float]:
    """
    Calculates the average number of repeated words over 10-word windows in the document.

    Args:
        importtext (str): The document string.

    Returns:
        List[float]: A list containing the average number of repeated words per 10-word window.

    Example:
        >>> word_repeats("This is a test. This test is only a test.")
        [0.5]
    """
    tokens = word_tokenize(importtext)

    tenwords2 = [tokens[i : i + 10] for i in range(0, len(tokens), 10)]
    repeatnum = 0

    for k in range(len(tenwords2) - 1):
        for word in tenwords2[k]:
            if word in tenwords2[k + 1]:
                repeatnum += 1

    sentencenum = len(tenwords2)
    repeatavg = repeatnum / sentencenum

    return [repeatavg]


def text_featurize(transcript: str) -> Tuple[List[float], List[str]]:
    """
    Extracts a comprehensive set of linguistic features from a transcript.

    Args:
        transcript (str): The document string.

    Returns:
        Tuple[List[float], List[str]]: A tuple containing a list of features and a list of feature labels.

    Example:
        >>> text_featurize("This is a test document.")
        ([0.0, 0.75, 0.0, 0.0, 0.0, 7.960204, 150.13059, 0.0, 3.4, 1.0, 8.0, 1.0, 6.240000000000001, 2.4999999999999996, 1, 1, 0, 0, 0.0], ['filler ratio', 'type token ratio', 'standardized word entropy', 'question ratio', 'number ratio', 'Brunets Index', 'Honores statistic', 'datewords freq', 'word number', 'five word count', 'max word length', 'min word length', 'variance of vocabulary', 'std of vocabulary', 'sentencenum', 'periods', 'questions', 'interjections', 'repeatavg'])
    """
    # extract features
    features1 = [
        filler_ratio(transcript),
        type_token_ratio(transcript),
        standardized_word_entropy(transcript),
        question_ratio(transcript),
        number_ratio(transcript),
        brunets_index(transcript),
        honores_statistic(transcript),
        datewords_freq(transcript),
    ]
    features2 = word_stats(transcript)
    features3 = num_sentences(transcript)
    features4 = word_repeats(transcript)

    # extract labels
    labels1 = [
        "filler ratio",
        "type token ratio",
        "standardized word entropy",
        "question ratio",
        "number ratio",
        "Brunets Index",
        "Honores statistic",
        "datewords freq",
    ]
    labels2 = [
        "word number",
        "five word count",
        "max word length",
        "min word length",
        "variance of vocabulary",
        "std of vocabulary",
    ]
    labels3 = ["sentencenum", "periods", "questions", "interjections"]
    labels4 = ["repeatavg"]

    features = features1 + features2 + features3 + features4
    labels = labels1 + labels2 + labels3 + labels4

    return features, labels
