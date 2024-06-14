import string


def word_count(docs, return_zero=[]):
    """

    Args:
            docs:
            return_zero: list containing docs not to count, default = None.
                    tokens

    Returns:

    """

    # initializing string
    # test_string = "Geeksforgeeks,    is best @# Computer Science Portal.!!!"
    word_counts = []
    for doc_i in docs:
        doc_i = str(doc_i)
        if doc_i in return_zero or len(doc_i) == 0:
            word_counts.append(0)
        else:
            # after asserting that len(doc_i)>0
            if len(doc_i) > 0 and sum([i.strip(string.punctuation).isalpha() for i in doc_i.split()]) == 0:
                # acronyms like 'R.I.P.'
                word_counts.append(1)
            else:
                wc_i = sum([i.strip(string.punctuation).isalpha() for i in doc_i.split()])
                word_counts.append(wc_i)
    if len(word_counts) == 1:
        return word_counts[0]
    else:
        return word_counts
