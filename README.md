# construct-tracker
Track and measure constructs, concepts or categories in text documents


# Installation

```bash
pip install construct-tracker
```

# Quick usage

# 1. Create a lexicon: keywords prototypically associated to a construct

We want to know if these documents contain mentions of certain construct "insight"

```python
documents = [
	'He is too competitive',
 	'Every time I speak with my cousin Bob, I have great moments of clarity and wisdom', # mention of insight
 	"He meditates a lot, but he's not super smart"] # only somewhat related to insight
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

Now count whether tokens appear in document:
```python
feature_vectors, matches_counter_d, matches_per_doc, matches_per_construct  = lexicon.extract(documents,
                                                                                      l.constructs,
                                                                                      normalize = False,
                                                                                      )
display(feature_vectors)
```
```
|   Insight |   word_count |
|----------:|-------------:|
|         0 |            4 |
|         2 |           17 |
|         0 |            8 |
```
The second document contains two matches related to Insight. This traditional approach is perfectly interpretable. Let's see which ones:
```python
print(matches_per_doc)
{0: {'Insight': (0, [])},
 1: {'Insight': (2, ['clarity', 'wisdom'])},
 2: {'Insight': (0, [])}}
```


# 2. Construct-text similarity: finding similar phrases to tokens in your lexicon
Lexicons may miss relevant words if not contained in the lexicon. Embeddings can find similar tokens. Vectorize lexicon tokens and document tokens (e.g., phrases) into embeddings. Compute similarity between both sets of tokens. Return maximum similarity. 

![Construct-Text similarity](docs/images/cts.pdf)

```python
feature_vectors_lc, cosine_scores_docs_lc = cts.measure(
      lexicon_dict = lexicon_dict,
      documents = documents,
    )
```






# Releasing a new version

- Create an API Token on [Pypi](https://pypi.org/).
- Create a [new release](https://github.com/danielmlow/construct-tracker/releases/new) on Github. 
Create a new tag in the form ``*.*.*``.


`pyproject.toml` has the requirements

<!-- tutorial to create package: https://www.youtube.com/watch?v=2goLiz4vTss -->

```
conda activate construct_poetry #created before
pip install poetry # create file with dependencies
poetry config virtualenvs.in-project true
poetry lock
poetry install
poetry config pypi-token.pypi API_token
poetry build
poetry publish
```

To reflect any new changes with a local installation
```
poetry build
pip install . 
```

To reflect any new changes in pypi, change the version number in pyproject.toml
```
poetry build
poetry publish --build
pip install --upgrade construct-tracker
```


Also, instead of modifying the `pyproject.toml` file by hand, you can use the `add` command. To automatically find a suitable version constraint and install the package and sub-dependencies.
```
poetry add spacy
```

### Steps to include dist/ in GitHub PR
- Commented out dist in `.gitignore`
- Pushed `dist/` contents (should include a .whl file) to GitHub
- Referenced .whl from `pip install` command above
- To generate a new .whl in your `dist` folder
- Run `poetry build`


# Pull requests

Daniel is merging into daniels_branch and then merging into main:

`git fetch origin pull/PULL_REQUEST_NUMBER/head:daniels_branch` pull PR, updates your local copy of the remote branches without changing your current working state

`git checkout daniels_branch` switch to branch. Make sure there are no errors. If there is a discrepancy between local and remote file, you can either commit your local files (with `add` and `commit`) or discard your local changes (`git checkout -- pyproject.toml`) or save them for later (`git stash`)

`git branch` check which branch you're on (the one with an * before it)

`git push origin daniels_branch`  push to branch

Then to make further commits, do:

`git checkout daniels_branch` to ensure the current branch is daniels_branch


```
git add .
git commit -m 'message'
git push --set-upstream origin daniels_branch
```

That will push your branch to the remote repository and set the upstream branch. After running this command, your local daniels_branch will be linked to the remote branch, and you can use git push and git pull commands without specifying the remote branch name in the future




## Testing updates and pull requests

The `@daniels_branch` installs that version. 

```
!pip install git+https://github.com/danielmlow/construct-tracker.git@daniels_branch#egg=construct_tracker&subdirectory=dist
```
