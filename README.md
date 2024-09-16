[![Build](https://github.com/danielmlow/construct-tracker/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/danielmlow/construct-tracker/actions/workflows/test.yaml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/danielmlow/construct-tracker/graph/badge.svg?token=9S8WY128PO)](https://codecov.io/gh/danielmlow/construct-tracker)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[![PyPI](https://img.shields.io/pypi/v/construct-tracker.svg)](https://pypi.org/project/construct-tracker/)
[![Python Version](https://img.shields.io/pypi/pyversions/construct-tracker)](https://pypi.org/project/construct-tracker)
[![License](https://img.shields.io/pypi/l/construct-tracker)](https://opensource.org/licenses/Apache-2.0)

<!-- [![pages](https://img.shields.io/badge/api-docs-blue)](https://sensein.github.io/construct-tracker) -->

# construct-tracker
Track and measure constructs, concepts or categories in text documents. Built on top of the litellm package to use most Generative AI models.

**If you use, please cite**: Low DM, Rankin O, Coppersmith DDL, Bentley KH, Nock MK, Ghosh SS (2024). Building lexicons with generative AI result in lightweight and interpretable text models with high content validity. arXiv.


# Installation

```bash
pip install construct-tracker
```

# Quick usage

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danielmlow/construct-tracker/blob/main/tutorials/construct_tracker.ipynb)

## Create a lexicon: keywords prototypically associated to a construct

We want to know if these documents contain mentions of certain construct "insight"

```python
documents = [
 	"Every time I speak with my cousin Bob, I have great moments of clarity and wisdom", # mention of insight
 	"He meditates a lot, but he's not super smart" # related to mindfulness, only somewhat related to insight
	"He is too competitive"] #not very related
```

Choose model [here](https://docs.litellm.ai/docs/providers) and obtain an API key from that provider. Cohere offers a free trial API key, 5 requests per minute. I'm going to choose GPT-4o:

```python
os.environ["OPENAI_API_KEY"]  = 'YOUR_OPENAI_API_KEY'
gpt4o = "gpt-4o-2024-05-13"
```

Two lines of code to create a lexicon
```python
l = lexicon.Lexicon()         # Initialize lexicon
l.add('Insight', section = 'tokens', value = 'create', source = gpt4o)
```

See results:
```python
print(l.constructs['Insight']['tokens'])
```
```
['acuity', 'acumen', 'analysis', 'apprehension', 'awareness', 'clarity', 'comprehension', 'contemplation', 'depth', 'discernment', 'enlightenment', 'epiphany', 'foresight', 'grasp', 'illumination', 'insightfulness', 'interpretation', 'introspection', 'intuition', 'meditation', 'perception', 'perceptiveness', 'perspicacity', 'profoundness', 'realization', 'recognition', 'reflection', 'revelation', 'shrewdness', 'thoughtfulness', 'understanding', 'vision', 'wisdom']
```

We'll repeat for other constructs ("Mindfulness", "Compassion"). Now count whether tokens appear in document:

```python
feature_vectors, matches_counter_d, matches_per_doc, matches_per_construct  = lexicon.extract(
	documents,
	l.constructs,
	normalize = False)

display(feature_vectors)
```

<img src="docs/images/counts.png" alt="Lexicon counts" width="700"/>

<!-- ```
|   Insight |   word_count |
|----------:|-------------:|
|         0 |            4 |
|         2 |           17 |
|         0 |            8 |
``` -->

This traditional approach is perfectly interpretable. The first document contains three matches related to insight. Let's see which ones with `highlight_matches()`:


```python
lexicon.highlight_matches(documents, 'Insight', matches_construct2doc, max_matches = 1)
```


<img src="docs/images/matches_insight.png" alt="Highlight matches" width="500"/>


<!-- ```python
print(matches_per_doc)
{0: {'Insight': (0, [])},
 1: {'Insight': (2, ['clarity', 'wisdom'])},
 2: {'Insight': (0, [])}}
``` -->
<br><br>



<!-- ## 2. Construct-text similarity (CTS): finding similar phrases to tokens in your lexicon

### Like Ctrl+F on steroids!
Lexicons may miss relevant words if not contained in the lexicon (it only counts exact matches). Embeddings can find semantically similar tokens. CTS will scan the document and return how similar is the most related phrase to any word in the lexicon.

<!-- magick -density 300 docs/images/cts.pdf -background white -alpha remove -quality 100 docs/images/cts.png -->
<!-- <img src="docs/images/cts.png" alt="Construct-text similarity" width="650"/> -->

<!-- It will vectorize lexicon tokens and document tokens (e.g., phrases) into embeddings (quantitivae vector representing aspects of meaning). Then it will compute the similarity between both sets of tokens and return the maximum similarity as its score for the document.  -->
<!--

```python
lexicon_dict = my_lexicon.to_dict()

features, documents_tokenized, lexicon_dict_final_order, cosine_similarities = cts.measure(
    lexicon_dict,
    documents,
    )

display(features)
```
<img src="docs/images/cts_scores.png" alt="Construct-text similarity" width="700"/>

So we see that even though compassion did not find an exact match it had some relationship to the first two documents.  -->



<!-- You can also sum the exact counts with the similarities for more fine-grained scores.

<img src="docs/images/cts_scores_sum.png" alt="Construct-text similarity" width="700"/> -->

We provide many features to add/remove tokens, generate definitions, validate with human ratings, and much more (see `tutorials/construct_tracker.ipynb`) [![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danielmlow/construct-tracker/blob/main/tutorials/construct_tracker.ipynb)

<br>

# Suicide Risk Lexicon

Lexicon is available in multiple formats:
- `https://github.com/danielmlow/construct-tracker/blob/main/src/construct_tracker/data/lexicons/suicide_risk_lexicon_v1-0/suicide_risk_lexicon_validated_24-08-02T21-27-35.csv`
- `https://github.com/danielmlow/construct-tracker/blob/main/src/construct_tracker/data/lexicons/suicide_risk_lexicon_v1-0/suicide_risk_lexicon_validated_24-08-02T21-27-35.json`

Or you can load lexicon object from the pickle file to extract features from new document.

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danielmlow/construct-tracker/blob/main/tutorials/suicide_risk_lexicon.ipynb)

We have created a lexicon with 49 risk factors for suicidal thoughts and behaviors validated by clinicians who are experts in suicide research.
```python
from construct_tracker import lexicon
# Load lexicon
srl = lexicon.load_lexicon(name = 'srl_v1-0')
# Load only tokens that are highly prototypical of each construct
srl_prototypes = lexicon.load_lexicon(name = 'srl_prototypes_v1-0')
```
<!-- lexicon_dict = srl.to_dict()
features, documents_tokenized, lexicon_dict_final_order, cosine_similarities = cts.measure(
    lexicon_dict,
    documents_subset,
    )

<img src="docs/images/srl_cts_scores.png" alt="Construct-text similarity of Suicide Risk Lexicon" width="700"/> -->


<br>

# Structure of the `lexicon.Lexicon()` object

```python
# Save general info on the lexicon
my_lexicon = lexicon.Lexicon()			# Initialize lexicon
my_lexicon.name = 'Insight'		# Set lexicon name
my_lexicon.description = 'Insight lexicon with constructs related to insight, mindfulness, and compassion'
my_lexicon.creator = 'DML' 				# your name or initials for transparency in logging who made changes
my_lexicon.version = '1.0'				# Set version. Over time, others may modify your lexicon, so good to keep track. MAJOR.MINOR. (e.g., MAJOR: new constructs or big changes to a construct, Minor: small changes to a construct)

# Each construct is a dict. You can save a lot of metadata depending on what you provide for each construct, for instance:
print(my_lexicon.constructs)
{
 'Insight': {
	'variable_name': 'insight', # a name that is not sensitive to case with no spaces
	'prompt_name': 'insight',
	'domain': 'psychology', 	 # to guide Gen AI model as to sense of the construct (depression has different senses in psychology, geology, and economics)
	'examples': ['clarity', 'enlightenment', 'wise'], # to guide Gen AI model
	'definition': "the clarity of understanding of one's thoughts, feelings and behavior", # can be used in prompt and/or human validation
	'definition_references': 'Grant, A. M., Franklin, J., & Langford, P. (2002). The self-reflection and insight scale: A new measure of private self-consciousness. Social Behavior and Personality: an international journal, 30(8), 821-835.',
	'tokens': ['acknowledgment',
	'acuity',
	'acumen',
	'analytical',
	'astute',
	'awareness',
	'clarity',
	...],
	'tokens_lemmatized': [], # when counting you can lemmatize all tokens for better results
	'remove': [], #which tokens to remove
	'tokens_metadata': {'gpt-4o-2024-05-13, temperature-0, ...': {
								'action': 'create',
								'tokens': [...],
								'prompt': 'Provide many single words and some short phrases ...',
								'time_elapsed': 14.21},
						{'gpt-4o-2024-05-13, temperature-1, ...': { ... }},
						}
	},
'Mindfulness': {...},
'Compassion': {...},
}
```

<!-- # Other features -->
<!-- TODO -->


# Contributing
<!-- TODO -->

See `docs/contributing.md`
