import requests
import json

PORT = 5000
IP = "0.0.0.0"

def make_request(prompt, num_tokens=1000, temperature=1.0, IP = IP, PORT = PORT):
    url = "http://" + str(IP) + ":" + str(PORT) + "/api"
    headers = {'Content-Type': 'application/json; charset=UTF-8'}
    print("NUM TOKENS IN API: ", num_tokens)
    data = {
        'prompts': [prompt],
        'tokens_to_generate': num_tokens,
        'temperature': temperature
        
    }

    response = requests.put(url, data=json.dumps(data), headers=headers)
    response_json = response.json()
    text = response_json["text"]
    return text

# Defaults
num_tokens = 100
temperature = 1.0

print("Welcome to interactive prompting. Write your prompt to the model below. Type ':q' to exit, :tokens x to specify number of tokens to generate, and :temperature y to specify the temperature.")
while True:
    user_input = input("> ")
    print("USER INPUT: " + str(user_input))

    if user_input.lower() == ":q":
        break
    elif ":tokens" in user_input.lower():
        splits = user_input.lower().split(" ")
        num_tokens = int(splits[1])
    elif ":temperature" in user_input.lower():
        splits = user_input.lower().split(" ")
        temperature = int(splits[1])
        
    else: 
        print("SENDING TO MODEL")
        text = make_request(user_input, num_tokens, temperature)
        print(text)