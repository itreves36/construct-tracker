def get_context(doc, token, n_words_pre=10, n_words_post=10):
    doc_pre_token = " ".join(doc.split(token)[0].split(" ")[-n_words_pre:])
    doc_post_token = " ".join(doc.split(token)[1].split(" ")[:n_words_post])
    doc_windowed = doc_pre_token + token + doc_post_token
    return doc_windowed


def get_docs_matching_token(docs, token, window=(10, 10), exact_match_n=4):
    docs_matching_token = [n for n in docs if token in n]
    if len(token) <= exact_match_n:
        # exact match
        docs_matching_token2 = docs_matching_token.copy()
        docs_matching_token = []
        for doc in docs_matching_token2:
            words = doc.split(" ")
            if token in words:
                docs_matching_token.append(doc)

    if window:
        docs_matching_token_windowed = []
        for doc in docs_matching_token:
            # doc = docs_matching_token[1]
            doc_windowed = get_context(doc, token, n_words_pre=window[0], n_words_post=window[1])
            docs_matching_token_windowed.append(doc_windowed)
        return docs_matching_token_windowed

    else:
        return docs_matching_token


# Example:
# get_docs_matching_token(['get paranoid and I think this is also a'],'thin', window = (10,10), exact_match_n = 4)
