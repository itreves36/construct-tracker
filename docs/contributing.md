


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
