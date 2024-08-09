# !pip install litellm==1.26.0


from litellm import completion
import litellm

litellm.drop_params=True # will ignore paramaters you set if they don't belong in a model

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

