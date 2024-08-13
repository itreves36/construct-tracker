
```python
feature_vectors_lc, cosine_scores_docs_lc = cts.measure(
      lexicon_dict = lexicon_dict
      documents = documents,
	  stored_embeddings = None,

    )
```

That function calls:
```python
feature_vectors_lc, cosine_scores_docs_lc = cts.measure(
      construct_tokens_d = construct_prototypes_tokens_d,
      construct_embeddings_d = construct_prototypes_embeddings_d,
      docs_embeddings_d = embeddings_tokens_docs_d,
      method = lexicon_clause, 
      summary_stat = ['max'],
      return_cosine_similarity=True,
      minmaxscaler = (0,1)
    )
```



TODO: describe these attributes and methods:

 'add',
 'attributes',
 'build',
 'clean_response',
 'construct_names',
 'constructs',
 'creator',
 'description',
 'exact_match_n',
 'exact_match_tokens',
 'extract',
 'generate_definition',
 'get_attribute',
 'name',
 'remove',
 'remove_from_all_constructs',
 'remove_tokens_containing_token',
 'save',
 'set_attribute',
 'to_dict',
 'to_pandas',
 'version'

