import os
import requests
import time
import argparse
import sys
import json

endpoint_id = os.environ["RUNPOD_ENDPOINT_ID"]
URI = f"https://api.runpod.ai/v2/{endpoint_id}/run"


def run(prompt, params, stream=False):
    request = {
        'prompt': prompt,
        'max_new_tokens': 1800,
        'temperature': 0.3,
        'top_k': 50,
        'top_p': 0.7,
        'repetition_penalty': 1.2,
        'batch_size': 8,
        'stream': stream
    }

    request.update(params)

    response = requests.post(URI, json=dict(input=request), headers={
        "Authorization": f"Bearer {os.environ['RUNPOD_AI_API_KEY']}"
    })

    if response.status_code == 200:
        data = response.json()
        task_id = data.get('id')
        return stream_output(task_id, stream=stream)


def stream_output(task_id, stream=False):
    url = f"https://api.runpod.ai/v2/{endpoint_id}/stream/{task_id}"
    headers = {
        "Authorization": f"Bearer {os.environ['RUNPOD_AI_API_KEY']}"
    }

    previous_output = ''

    try:
        while True:
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                data = response.json()

                if len(data['stream']) > 0:
                    new_output = data['stream'][0]['output']

                    if stream:
                        print(new_output, end ="")

                    previous_output += new_output
                
                if data.get('status') == 'COMPLETED':
                    if not stream:
                        return previous_output
                    break
                    
            elif response.status_code >= 400:
                print(response)

            # Sleep for 0.1 seconds between each request
            time.sleep(0.1 if stream else 1)
    except Exception as e:
        print(e)
        cancel_task(task_id)
    

def cancel_task(task_id):
    url = f"https://api.runpod.ai/v2/{endpoint_id}/cancel/{task_id}"
    headers = {
        "Authorization": f"Bearer {os.environ['RUNPOD_AI_API_KEY']}"
    }
    response = requests.get(url, headers=headers)
    return response


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runpod AI CLI')
    parser.add_argument('-s', '--stream', action='store_true', help='Stream output')
    parser.add_argument('-p', '--params_json', type=str, help='JSON string of generation params')

    prompt = '''<s>[INST] Give me 10 unique baby names.
return the results in JSON Format
Strictly follow the JSON Schema below:
{
    "baby_names": [
        "baby_name1",
        "baby_name2",
        "baby_name3",
        "baby_name4",
        "baby_name5",
        "baby_name6",
        "baby_name7",
        "baby_name8",
        "baby_name9",
        "baby_name10",
    ]
}
Only return the JSON results and nothing else.[/INST]\n'''

    args = parser.parse_args()
    params = json.loads(args.params_json) if args.params_json else {}
    start = time.time()
    print(run(prompt, params=params, stream=args.stream))
    print("Time taken: ", time.time() - start, " seconds")
