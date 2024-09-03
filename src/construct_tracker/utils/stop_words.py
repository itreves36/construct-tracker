"""Remove stop words."""

import string
from typing import List, Optional

import nltk
from nltk.corpus import stopwords

from .tokenizer import spacy_tokenizer

nltk_language = {"en": "English", "es": "Spanish"}

nltk_stop_words = {
    "en": [
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "you're",
        "you've",
        "you'll",
        "you'd",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "she's",
        "her",
        "hers",
        "herself",
        "it",
        "it's",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "that'll",
        "these",
        "those",
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "a",
        "an",
        "the",
        "and",
        "but",
        "if",
        "or",
        "because",
        "as",
        "until",
        "while",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "s",
        "t",
        "can",
        "will",
        "just",
        "don",
        "don't",
        "should",
        "should've",
        "now",
        "d",
        "ll",
        "m",
        "o",
        "re",
        "ve",
        "y",
        "ain",
        "aren",
        "aren't",
        "couldn",
        "couldn't",
        "didn",
        "didn't",
        "doesn",
        "doesn't",
        "hadn",
        "hadn't",
        "hasn",
        "hasn't",
        "haven",
        "haven't",
        "isn",
        "isn't",
        "ma",
        "mightn",
        "mightn't",
        "mustn",
        "mustn't",
        "needn",
        "needn't",
        "shan",
        "shan't",
        "shouldn",
        "shouldn't",
        "wasn",
        "wasn't",
        "weren",
        "weren't",
        "won",
        "won't",
        "wouldn",
        "wouldn't",
    ],
    "es": [
        "de",
        "la",
        "que",
        "el",
        "en",
        "y",
        "a",
        "los",
        "del",
        "se",
        "las",
        "por",
        "un",
        "para",
        "con",
        "no",
        "una",
        "su",
        "al",
        "lo",
        "como",
        "más",
        "pero",
        "sus",
        "le",
        "ya",
        "o",
        "este",
        "sí",
        "porque",
        "esta",
        "entre",
        "cuando",
        "muy",
        "sin",
        "sobre",
        "también",
        "me",
        "hasta",
        "hay",
        "donde",
        "quien",
        "desde",
        "todo",
        "nos",
        "durante",
        "todos",
        "uno",
        "les",
        "ni",
        "contra",
        "otros",
        "ese",
        "eso",
        "ante",
        "ellos",
        "e",
        "esto",
        "mí",
        "antes",
        "algunos",
        "qué",
        "unos",
        "yo",
        "otro",
        "otras",
        "otra",
        "él",
        "tanto",
        "esa",
        "estos",
        "mucho",
        "quienes",
        "nada",
        "muchos",
        "cual",
        "poco",
        "ella",
        "estar",
        "estas",
        "algunas",
        "algo",
        "nosotros",
        "mi",
        "mis",
        "tú",
        "te",
        "ti",
        "tu",
        "tus",
        "ellas",
        "nosotras",
        "vosotros",
        "vosotras",
        "os",
        "mío",
        "mía",
        "míos",
        "mías",
        "tuyo",
        "tuya",
        "tuyos",
        "tuyas",
        "suyo",
        "suya",
        "suyos",
        "suyas",
        "nuestro",
        "nuestra",
        "nuestros",
        "nuestras",
        "vuestro",
        "vuestra",
        "vuestros",
        "vuestras",
        "esos",
        "esas",
        "estoy",
        "estás",
        "está",
        "estamos",
        "estáis",
        "están",
        "esté",
        "estés",
        "estemos",
        "estéis",
        "estén",
        "estaré",
        "estarás",
        "estará",
        "estaremos",
        "estaréis",
        "estarán",
        "estaría",
        "estarías",
        "estaríamos",
        "estaríais",
        "estarían",
        "estaba",
        "estabas",
        "estábamos",
        "estabais",
        "estaban",
        "estuve",
        "estuviste",
        "estuvo",
        "estuvimos",
        "estuvisteis",
        "estuvieron",
        "estuviera",
        "estuvieras",
        "estuviéramos",
        "estuvierais",
        "estuvieran",
        "estuviese",
        "estuvieses",
        "estuviésemos",
        "estuvieseis",
        "estuviesen",
        "estando",
        "estado",
        "estada",
        "estados",
        "estadas",
        "estad",
        "he",
        "has",
        "ha",
        "hemos",
        "habéis",
        "han",
        "haya",
        "hayas",
        "hayamos",
        "hayáis",
        "hayan",
        "habré",
        "habrás",
        "habrá",
        "habremos",
        "habréis",
        "habrán",
        "habría",
        "habrías",
        "habríamos",
        "habríais",
        "habrían",
        "había",
        "habías",
        "habíamos",
        "habíais",
        "habían",
        "hube",
        "hubiste",
        "hubo",
        "hubimos",
        "hubisteis",
        "hubieron",
        "hubiera",
        "hubieras",
        "hubiéramos",
        "hubierais",
        "hubieran",
        "hubiese",
        "hubieses",
        "hubiésemos",
        "hubieseis",
        "hubiesen",
        "habiendo",
        "habido",
        "habida",
        "habidos",
        "habidas",
        "soy",
        "eres",
        "es",
        "somos",
        "sois",
        "son",
        "sea",
        "seas",
        "seamos",
        "seáis",
        "sean",
        "seré",
        "serás",
        "será",
        "seremos",
        "seréis",
        "serán",
        "sería",
        "serías",
        "seríamos",
        "seríais",
        "serían",
        "era",
        "eras",
        "éramos",
        "erais",
        "eran",
        "fui",
        "fuiste",
        "fue",
        "fuimos",
        "fuisteis",
        "fueron",
        "fuera",
        "fueras",
        "fuéramos",
        "fuerais",
        "fueran",
        "fuese",
        "fueses",
        "fuésemos",
        "fueseis",
        "fuesen",
        "sintiendo",
        "sentido",
        "sentida",
        "sentidos",
        "sentidas",
        "siente",
        "sentid",
        "tengo",
        "tienes",
        "tiene",
        "tenemos",
        "tenéis",
        "tienen",
        "tenga",
        "tengas",
        "tengamos",
        "tengáis",
        "tengan",
        "tendré",
        "tendrás",
        "tendrá",
        "tendremos",
        "tendréis",
        "tendrán",
        "tendría",
        "tendrías",
        "tendríamos",
        "tendríais",
        "tendrían",
        "tenía",
        "tenías",
        "teníamos",
        "teníais",
        "tenían",
        "tuve",
        "tuviste",
        "tuvo",
        "tuvimos",
        "tuvisteis",
        "tuvieron",
        "tuviera",
        "tuvieras",
        "tuviéramos",
        "tuvierais",
        "tuvieran",
        "tuviese",
        "tuvieses",
        "tuviésemos",
        "tuvieseis",
        "tuviesen",
        "teniendo",
        "tenido",
        "tenida",
        "tenidos",
        "tenidas",
        "tened",
    ],
}


def remove_punctuation(doc: str) -> str:
    """
    Removes all punctuation from the input document.

    Args:
        doc (str): The document from which to remove punctuation.

    Returns:
        str: The document with punctuation removed.

    Example:
        >>> remove_punctuation("Hello, world!")
        'Hello world'
    """
    doc = doc.translate(str.maketrans("", "", string.punctuation))
    return doc


def return_stopwords(method: str = "nltk", language: str = "en") -> List[str]:
    """
    Returns a list of stopwords for the specified language using the specified method.

    Args:
        method (str, optional): The method to use for retrieving stopwords. Defaults to "nltk".
        language (str, optional): The language for which to retrieve stopwords. Defaults to "en".

    Returns:
        List[str]: A list of stopwords.

    Example:
        >>> return_stopwords()
        ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', ...]
    """
    if method == "nltk":
        return stopwords.words(nltk_language.get(language))


def remove_stopwords_doc(
    word_list: List[str], method: str = "nltk", language: str = "en", extend_stopwords: Optional[List[str]] = None
) -> str:
    """
    Removes stopwords from a list of words.

    Args:
        word_list (List[str]): A list of words to filter.
        method (str, optional): The method to use for retrieving stopwords. Defaults to "nltk".
        language (str, optional): The language for which to retrieve stopwords. Defaults to "en".
        extend_stopwords (Optional[List[str]], optional): Additional stopwords to include. Defaults to None.

    Returns:
        str: A string with stopwords removed.

    Example:
        >>> remove_stopwords_doc(["this", "is", "a", "test"])
        'test'
    """
    if method == "nltk":
        sws = nltk_stop_words.get(language)
        if sws is None:
            try:
                nltk.download("stopwords")
            except Exception as e:
                print(f"An error occurred while downloading NLTK stopwords: {e}")
                return " ".join(word_list)

            sws = stopwords.words(nltk_language.get(language))

        if extend_stopwords:
            sws.extend(extend_stopwords)

        filtered_words = [word for word in word_list if word not in sws]
    filtered_words = " ".join(filtered_words)
    return filtered_words


def remove(
    docs: List[str],
    language: str = "en",
    method: str = "nltk",
    remove_punct: bool = True,
    extend_stopwords: Optional[List[str]] = None,
) -> List[str]:
    """
    Preprocesses a list of documents by removing punctuation, converting to lowercase, tokenizing, and removing stopwords.

    Args:
        docs (List[str]): A list of documents to preprocess.
        language (str, optional): The language of the documents. Defaults to "en".
        method (str, optional): The method to use for retrieving stopwords. Defaults to "nltk".
        remove_punct (bool, optional): Whether to remove punctuation. Defaults to True.
        extend_stopwords (Optional[List[str]], optional): Additional stopwords to include. Defaults to None.

    Returns:
        List[str]: A list of preprocessed documents.

    Example:
        >>> docs = ["This is a test document.", "I love programming!"]
        >>> remove(docs)
        ['test document', 'love programming']
    """
    if remove_punct:
        docs = [remove_punctuation(doc) for doc in docs]

    docs = [doc.lower() for doc in docs]
    docs = spacy_tokenizer(docs, method="word", language="en", lowercase=True)

    filtered_docs = [
        remove_stopwords_doc(word_list, method=method, language=language, extend_stopwords=extend_stopwords)
        for word_list in docs
    ]
    return filtered_docs
