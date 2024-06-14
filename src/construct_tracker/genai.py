# !pip install litellm==1.26.0


from litellm import completion


def api_request(
    prompt,
    model="commmand-nightly",
    api_key=None,
    temperature=0.1,
    top_p=1,
    timeout=45,
    num_retries=2,
    max_tokens=None,
    seed=None,
    response_format=None,
):
    # Open AI status: https://status.openai.com/

    messages = [{"content": prompt, "role": "user"}]
    responses = completion(
        model=model,
        messages=messages,
        api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        timeout=timeout,
        num_retries=num_retries,
        max_tokens=max_tokens,
        seed=seed,
        # response_format = response_format
    )
    response = responses.get("choices")[0].get("message").get("content")  # access response for first message
    return response


'''

# https://github.com/ollama/ollama/blob/main/docs/faq.md#does-ollama-send-my-prompts-and-answers-back-to-ollamacom
# !pip install ollama==0.1.6
import ollama


# ollama.pull('gemma:2b')

response = ollama.chat(model='gemma:7b', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
print(response['message']['content'])

# TODO: implement predict_proba: https://github.com/skorch-dev/skorch/blob/f3d9ea367515c8bce3d48bebed231185c54c4edf/skorch/llm/classifier.py#L392
# TODO: you might be able to use ollama if you're running the server locally: https://litellm.vercel.app/docs/provide


# How to parse output for classification

# https://github.com/skorch-dev/skorch/blob/master/skorch/llm/classifier.py


import api_keys
reload(api_keys)
os.environ["TOGETHERAI_API_KEY"] = api_keys.togetherai
os.environ["OPENAI_API_KEY"] = api_keys.open_ai


import prompts
reload(prompts)




input_dir = './data/input/'



# prompt_template = prompts.ZERO_SHOT_CLF_PROMPT_TEMPLATE
prompt_template = prompts.DEFAULT_PROMPT_ZERO_SHOT

prompt_template = """You are a text classification assistant.

The text to classify:

```
{x}
```

Assign a probability for each possible label: '{labels[0]}' or '{labels[1]}' and return in a JSON format

For instance, return this in JSON format:
'{labels[0]}': <your_score>,
'{labels[1]}': <your_score>

Do not provide additional text or explanations, just the JSON output.
"""
# Your response (only return the label without additional text):


labels = ['loneliness', 'not loneliness']
X = ["No one cares about me", "I have many friends", "The table is round"]

models = ['gpt-4-1106-preview','together_ai/meta-llama/Llama-2-7b-chat-hf','together_ai/meta-llama/Llama-2-13b-chat-hf', 'together_ai/meta-llama/Llama-2-70b-chat-hf']

model = models[1]
print(model)

for x in X:
	prompt = prompt_template.format(labels = labels,x=x)
	print(prompt)
	response = api_request(model = model, prompt = prompt)
	print(response)


find_json_in_string(response)

# prompt = 'Hello'


# !pip install --upgrade gpt4all

import sys
sys.path.append("/Users/danielmlow/Dropbox (MIT)/datum/concept-tracker/gpt4all/gpt4all-bindings/python")
from gpt4all import GPT4All
model = GPT4All("orca-mini-3b-gguf2-q4_0.gguf")
output = model.generate("The capital of France is ", max_tokens=3)
print(output)



def find_json_in_string(string: str) -> str:
    """Finds the JSON object in a string.

    Parameters
    ----------
    string : str
        The string to search for a JSON object.

    Returns
    -------
    json_string : str
    """
    start = string.find("{")
    end = string.rfind("}")
    if start != -1 and end != -1:
        json_string = string[start : end + 1]
    else:
        json_string = "{}"
    return json_string

response_json_str = find_json_in_string(response)
print(response_json_str)
import json
as_json = json.loads(response_json_str)
as_json


# gpt4all
# ============================================================================================================
# models: https://raw.githubusercontent.com/nomic-ai/gpt4all/main/gpt4all-chat/metadata/models2.json
!pip install gpt4all==0.3.0
!pip install --upgrade gpt4all

import gpt4all


from gpt4all import GPT4All
model = GPT4All("orca-mini-3b-gguf2-q4_0.gguf")
output = model.generate(prompt, max_tokens=10)
print(output)

!pip uninstall -y gpt4all




# ============================================================================================================
# Import the necessary modules
from skllm.datasets import get_classification_dataset
from skllm.config import SKLLMConfig
from skllm.models.gpt.classification.zero_shot import ZeroShotGPTClassifier

# Configure the credentials
SKLLMConfig.set_openai_key(api_keys.open_ai)
# SKLLMConfig.set_openai_org("<YOUR_ORGANIZATION_ID>")

# Load a demo dataset
X, y = get_classification_dataset() # labels: positive, negative, neutral

# Initialize the model and make the predictions
# !pip install openai==1.0
import openai

openai.__version__
clf = ZeroShotGPTClassifier(model="gpt-4")
clf.fit(X,y)
clf.predict(X)








model = "meta-llama/Llama-2-7b-hf"
messages = [{"role": "user", "content": "Hey, how's it going?"}] # LiteLLM follows the OpenAI format
api_base = "https://ag3dkq4zui5nu8g3.us-east-1.aws.endpoints.huggingface.cloud"

### CALLING ENDPOINT
completion(model=model, messages=messages, custom_llm_provider="huggingface", api_base=api_base)



from litellm import completion

model = "meta-llama/Llama-2-7b-hf"
messages = [{"role": "user", "content": "Hey, how's it going?"}] # LiteLLM follows the OpenAI format
api_base = "https://ag3dkq4zui5nu8g3.us-east-1.aws.endpoints.huggingface.cloud"

### CALLING ENDPOINT
completion(model=model, messages=messages, custom_llm_provider="huggingface", api_base=api_base)



model = "deepset/deberta-v3-large-squad2"
messages = [{"role": "user", "content": "Hey, how's it going?"}] # LiteLLM follows the OpenAI format

### CALLING ENDPOINT
completion(model=model, messages=messages, custom_llm_provider="huggingface")


messages = [{ "content": "There's a llama in my garden 😱 What should I do?","role": "user"}]

# e.g. Call 'WizardLM/WizardCoder-Python-34B-V1.0' hosted on HF Inference endpoints
response = completion(
  model="huggingface/meta-llama/Llama-2-7b-chat",
  messages=messages,
#   api_base="https://my-endpoint.huggingface.cloud"
)

print(response)
api_request(model = "huggingface/meta-llama/Llama-2-7b-chat", prompt="There's a llama in my garden 😱 What should I do?")


!pip install skorch
from skorch.llm import ZeroShotClassifier

clf = ZeroShotClassifier('meta-llama/Llama-2-7b')
clf.fit(X=None, y=['positive', 'negative'])

review = """I'm very happy with this new smartphone. The display is excellent
and the battery lasts for days. My only complaint is the camera, which could
be better. Overall, I can highly recommend this product."""

clf.predict([review])  # returns 'positive'
clf.predict_proba([review])  # returns array([[0.05547452, 0.94452548]])
'''
