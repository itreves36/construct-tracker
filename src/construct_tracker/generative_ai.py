from litellm import completion


def api_request(prompt, model="gtp-4", api_key=None):
    messages = [{"content": prompt, "role": "user"}]

    responses = completion(model=model, messages=messages, api_key=api_key)
    response = responses.get("choices")[0].get("message").get("content")  # access response for first message
    return response
