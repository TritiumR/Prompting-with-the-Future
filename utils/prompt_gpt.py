import requests
import os
import numpy as np
import base64


# OpenAI API Key
# api_key = None
api_key = "sk-_a66PUCIY5_DkEtoqIq3fcyBHeCKiXIkE_sTMNXOi-T3BlbkFJRE0XNZGEmXDxPDo6AulxjojmloSLFB4ynom7I4AKoA"
headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}


def get_toss(content):
    toss = content.split('Success:')[-1].strip(' :*.').lower()
    assert toss in ['yes', 'no']
    return toss == 'yes'


def write_reward_function(content, file_name):
    with open('prompts/reward_template.py', "r") as f:
        template = f.read()

    for line in content.split('\n'):
        template += f"    {line.strip()}\n"
    
    template += "    criteria = dict()\n"
    template += "    try: \n"
    template += "        criteria['gripper_distance'] = gripper_distance\n"
    template += "    except: \n"
    template += "        pass\n"
    template += "    try: \n"
    template += "        criteria['target_distance'] = target_distance\n"
    template += "    except: \n"
    template += "        pass\n"
    template += "    try: \n"
    template += "        criteria['rotation_distance'] = rotation_distance\n"
    template += "    except: \n"
    template += "        pass\n"
    template += "    return reward, criteria\n"
    
    with open(file_name, "w") as f:
        f.write(template)
    

def get_answer_abcd(content):
    letter = content.split('Best Result:')[-1].split('Confidence:')[0].strip(' :*.')
    assert letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    if letter == 'A':
        return 1
    elif letter == 'B':
        return 2
    elif letter == 'C':
        return 3
    elif letter == 'D':
        return 4
    elif letter == 'E':
        return 5
    elif letter == 'F':
        return 6
    elif letter == 'G':
        return 7
    elif letter == 'H':
        return 8

def get_answer(content):
    return int(content.split('Best Result:')[-1].split('Confidence:')[0].strip(' :*.'))

def get_view(content):
    return int(content.split('Best View:')[-1].strip(' :*.'))

def get_stage(content):
    return int(content.split('Current Stage:')[-1].strip(' :*.'))

def get_confidence(content):
    return float(content.split('Best Result:')[-1].split('Confidence:')[-1].strip(' :*.'))

def get_change(content):
    answer = content.split('Change View:')[-1].strip(' :*.').lower()
    assert 'yes' == answer or 'no' == answer
    return 'yes' == answer

def get_grasp(content):
    answer = content.split('Grasp:')[-1].strip(' :*.').lower()
    assert 'yes' == answer or 'no' == answer
    return 'yes' == answer

def get_release(content):
    answer = content.split('Release:')[-1].strip(' :*.').lower()
    assert 'yes' == answer or 'no' == answer
    return 'yes' == answer

def get_success(content):
    answer = content.split('Satisfied:')[-1].strip(' :*.').lower()
    assert 'yes' == answer or 'no' == answer
    return 'yes' == answer

def get_close_gripper(content):
    answer = content.split('Keep Gripper Closed:')[-1].strip(' :*.').lower()
    assert 'yes' == answer or 'no' == answer
    return 'yes' == answer

def get_subgoals(content):
    print(content)

    subgoal_list = []
    goal_id = 0
    content = content.split('Sub Goals:')[-1]
    while True:
        goal_id += 1
        goal = content.split(f'{goal_id}.')[-1].split(f'{goal_id + 1}.')[0].strip(' :*.\n"')
        subgoal_list.append(goal)
        if f'{goal_id + 1}.' not in content:
            break

    # content = content.split('[')[-1].split(']')[0].strip(' :*."')
    # subgoal_list = content.split(',')
    return subgoal_list

def get_names(content):
    print(content)

    content = content.split('Objects:')[-1]

    name_list = []
    name_id = 1
    while True:
        if f'{name_id}.' not in content:
            break
        name = content.split(f'{name_id}.')[-1].split(f'{name_id + 1}.')[0].strip(' :*.\n"')
        name_list.append(name)
        name_id += 1

    # content = content.split('[')[-1].split(']')[0].strip(' :*."')
    # subgoal_list = content.split(',')
    return name_list

def get_description_list(content, num_results):
    description_list = []
    for idx in range(1, num_results + 1):
        start = content.find(f'Description {idx}:')
        end = content.find(f'Description {idx + 1}:')
        if idx == num_results:
            end = len(content)
        if start == -1 or end == -1:
            print('Description not found')
            return None
        description = content[start:end]
        description_list.append(description)
    return description_list


def get_action(content):
    return content.split('Best Action:')[-1].strip(' :*."')


def generate_description(results: list, system_prompt: str, model: str = 'gpt-4o',
                        temperature: float = 0.0, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                        presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []
    for idx, result in enumerate(results):
        usr_content.append({"type": "text", "text": f"This is the multi-view images of result {idx + 1}. The views are 'top-down', 'side', 'front', 'side of gripper', 'front of gripper', and 'wrist camera'."})
        for image in result:
            usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # temperature=temperature,
        # max_tokens=max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    print(response)
    content = response['choices'][0]['message']['content']

    description_list = get_description_list(content, len(results))

    return content


def seperate_generate_description(results: list, system_prompt: str, motion_name_list: str = None, model: str = 'gpt-4o',
                        temperature: float = 0.0, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                        presence_penalty: float = 0.0, stop: list = None ):
    description_list = []
    for idx, result in enumerate(results):
        usr_content = []
        usr_content.append({"type": "text", "text": "These are the multi-view images of the result. The views are 'top-down', 'side', 'front', 'side of gripper', 'front of gripper', and 'wrist camera'."})
        if motion_name_list is not None:
            usr_content.append({"type": "text", "text": f'The motion direction relative to the robot base is {motion_name_list[idx]}'})
        for image in result:
            usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})
        
        payload = {
            "model" : f"{model}",
            "messages" : [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": usr_content},
            ],
            # temperature=temperature,
            # max_tokens=max_tokens,
            # top_p=top_p,
            # frequency_penalty=frequency_penalty,
            # presence_penalty=presence_penalty,
            # stop=stop,
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
        print(response)
        content = response['choices'][0]['message']['content']
        description_list.append(content)
        
    return description_list


def generate_description_concatenate(results: list, system_prompt: str, motion_name_list: str = None, model: str = 'gpt-4o',
                        temperature: float = 0.0, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                        presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []
    for idx, result in enumerate(results):
        usr_content.append({"type": "text", "text": "These are the multi-view images of the result. The views are 'top-down', 'side', 'front', 'side of gripper', 'front of gripper', and 'wrist camera' (From left-to-right then up-to-down)."})
        if motion_name_list is not None:
            usr_content.append({"type": "text", "text": f'The motion direction relative to the robot base is {motion_name_list[idx]}'})
        concatenated_image = result
        usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{concatenated_image}", "detail": "high"}})
        
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # temperature=temperature,
        # max_tokens=max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content


def seperate_generate_description_concatenate(results: list, system_prompt: str, motion_name_list: str = None, model: str = 'gpt-4o',
                        temperature: float = 0.0, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                        presence_penalty: float = 0.0, stop: list = None ):
    description_list = []
    for idx, result in enumerate(results):
        usr_content = []
        usr_content.append({"type": "text", "text": "These are the multi-view images of the result. The views are 'top-down', 'side', 'front', 'side of gripper', 'front of gripper', and 'wrist camera' (From left-to-right then up-to-down)."})
        if motion_name_list is not None:
            usr_content.append({"type": "text", "text": f'The motion direction in the world coordinate is {motion_name_list[idx]}'})
        concatenated_image = result
        usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{concatenated_image}", "detail": "high"}})
        
        payload = {
            "model" : f"{model}",
            "messages" : [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": usr_content},
            ],
            # temperature=temperature,
            # max_tokens=max_tokens,
            # top_p=top_p,
            # frequency_penalty=frequency_penalty,
            # presence_penalty=presence_penalty,
            # stop=stop,
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
        print(response)
        content = response['choices'][0]['message']['content']
        # print(content)
        description_list.append(content)
        
    return description_list


def generate_choice(results: list, descriptions: list, system_prompt: str, history = None, model: str = 'gpt-4o',
                    temperature: float = 0.0, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                    presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []
    for idx, result in enumerate(results):
        usr_content.append({"type": "text", "text": f"These are the multi-view images of result {idx + 1}. The views are 'top-down', 'side', 'front', 'side of gripper', 'front of gripper', and 'wrist camera'."})
        usr_content.append({"type": "text", "text": f'This is the description of result {idx + 1} : {descriptions[idx]}'})
        for image in result:
            usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # temperature=temperature,
        # max_tokens=max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content


def generate_choice_concatenate(results: list, descriptions: list, system_prompt: str, motion_name_list: str = None, history = None, model: str = 'gpt-4o',
                    temperature: float = 0.0, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                    presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []
    for idx, result in enumerate(results):
        usr_content.append({"type": "text", "text": f"These are the multi-view images of result {idx + 1}. The views are 'top-down', 'side', 'front', 'side of gripper', 'front of gripper', and 'wrist camera' (From left-to-right then up-to-down)."})
        if motion_name_list is not None:
            usr_content.append({"type": "text", "text": f'The motion direction relative to the robot base is {motion_name_list[idx]}'})
        usr_content.append({"type": "text", "text": f'This is the description of result {idx + 1} : {descriptions[idx]}'})
        concatenated_image = result
        usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{concatenated_image}", "detail": "high"}})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # temperature=temperature,
        # max_tokens=max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content


def generate_response(results: list, system_prompt: str, history = None, model: str = 'gpt-4o',
                      temperature: float = 0.0, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                      presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []
    for idx, result in enumerate(results):
        usr_content.append({"type": "text", "text": f"These are the multi-view images of result {idx + 1}. The views are 'top-down', 'side', 'front', 'side of gripper', 'front of gripper', and 'wrist camera'."})
        for image in result:
            usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # temperature=temperature,
        # max_tokens=max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content


def generate_response_concatenate(results: list, system_prompt: str, motion_name_list: list = None, history = None, model: str = 'gpt-4o',
                      temperature: float = 0.0, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                      presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []
    for idx, result in enumerate(results):
        usr_content.append({"type": "text", "text": f"These are the multi-view images of result {idx + 1}. The views are 'top-down', 'side', 'front', 'side of gripper', 'front of gripper', and 'wrist camera' (From left-to-right then up-to-down)."})
        if motion_name_list is not None:
            usr_content.append({"type": "text", "text": f'The motion direction relative to the robot base is {motion_name_list[idx]}'})
        concatenated_image = result
        usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{concatenated_image}", "detail": "high"}})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # temperature=temperature,
        # max_tokens=max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content

def generate_action_concatenate(image: str, system_prompt: str, history = None, model: str = 'gpt-4o',
                      temperature: float = 0.0, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                      presence_penalty: float = 0.0, stop: list = None ):
    messages = [{"role": "system", "content": system_prompt}]
    if history is not None:
        history_images = history[0]
        history_contents = history[1]
        print('lengh of history: ', len(history_images))
        for history_image, history_content in zip(history_images, history_contents):
            history_usr_content = []
            history_usr_content.append({"type": "text", "text": f"This is the image containing multi-view observations of the current state."})
            history_usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{history_image}", "detail": "high"}})
            messages.append({"role": "user", "content": history_usr_content})
            messages.append({"role": "assistant", "content": history_content})
    
    usr_content = []
    usr_content.append({"type": "text", "text": f"This is the image containing multi-view observations of the current state."})
    usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})
    
    messages.append({"role": "user", "content": usr_content})
    payload = {
        "model" : f"{model}",
        "messages" : messages,
        # temperature=temperature,
        # max_tokens=max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content

def per_image_generate_description(image: str, system_prompt: str, view_name: str, motion_name: str = None, model: str = 'gpt-4o',
                      temperature: float = 0.0, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                      presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []
    usr_content.append({"type": "text", "text": f'This is the {view_name} view observation of the result.'})
    if motion_name is not None:
        usr_content.append({"type": "text", "text": f'The motion direction in the world coordinate is {motion_name}.'})
    usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # temperature=temperature,
        # max_tokens=max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content

def per_action_generate_description(per_view_contents: list, system_prompt: str, view_name_list: str, motion_name: str = None, model: str = 'gpt-4o',
                      temperature: float = 0.0, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                      presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []
    if motion_name is not None:
        usr_content.append({"type": "text", "text": f'The motion direction in the world coordinate is {motion_name}.'})
    for idx, per_view_content in enumerate(per_view_contents):
        usr_content.append({"type": "text", "text": f'This is the {view_name_list[idx]} view observation of the result: {per_view_content}'})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # temperature=temperature,
        # max_tokens=max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content


def no_image_generate_choice(descriptions: list, system_prompt: str, motion_name_list: list = None, history = None, model: str = 'gpt-4o',
                    temperature: float = 0.0, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                    presence_penalty: float = 0.0, stop: list = None ):
    messages = [{"role": "system", "content": system_prompt}]
    if history is not None:
        history_descriptions_list = history[0]
        history_contents = history[1]
        print('lengh of history: ', len(history_descriptions))
        for history_descriptions, history_content in zip(history_descriptions_list, history_contents):
            history_usr_content = []
            for idx, history_description in enumerate(history_descriptions):
                if motion_name_list is not None:
                    history_usr_content.append({"type": "text", "text": f'The motion direction in the world coordinate is {motion_name_list[idx]}.'})
                history_usr_content.append({"type": "text", "text": f"This is the description of result {idx + 1}: {history_description}"})
            messages.append({"role": "user", "content": history_usr_content})
            messages.append({"role": "assistant", "content": history_content})

    usr_content = []
    for idx, description in enumerate(descriptions):
        if motion_name_list is not None:
            usr_content.append({"type": "text", "text": f'The motion direction in the world coordinate is {motion_name_list[idx]}.'})
        usr_content.append({"type": "text", "text": f"This is the description of result {idx + 1}: {description}"})

    messages.append({"role": "user", "content": usr_content})
    
    payload = {
        "model" : f"{model}",
        "messages" : messages,
        # temperature=temperature,
        # max_tokens=max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content


def simple_generate_response(results: list, system_prompt: str, history = [], grasping = False, model: str = 'gpt-4o',
                      temperature: float = 0.0, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                      presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []
    if len(history) > 0:
        usr_content.append({"type": "text", "text": f"These are history obervations of the robot:"})
        for idx, history_image in enumerate(history):
            usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{history_image}", "detail": "high"}})
        usr_content.append({"type": "text", "text": f"The following are the results of the sampled actions you need to choose from."})

    if grasping:
        usr_content.append({"type": "text", "text": f'The gripper is grasping the object now.'})

    for idx, result in enumerate(results):
        usr_content.append({"type": "text", "text": f"This is the obervation of future result {idx + 1}:"})
        image = result[0]
        usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # careful!!! temperature
        # "temperature" : temperature,
        # max_tokens=max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content


def simple_generate_response_ABCD(results: list, system_prompt: str, history = [], model: str = 'gpt-4o',
                      temperature: float = 0.0, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                      presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []
    # if len(history) > 0:
    #     usr_content.append({"type": "text", "text": f"These are history obervations of the robot:"})
    #     for idx, history_image in enumerate(history):
    #         usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{history_image}", "detail": "high"}})
    #     usr_content.append({"type": "text", "text": f"The following are the results of the sampled actions you need to choose from."})

    usr_content.append({"type": "text", "text": f"These are the obervations of four future results:"})
    image = results[0]
    usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # temperature=temperature,
        # max_tokens=max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content


def simple_generate_response_abcd(results: list, system_prompt: str, history = [], model: str = 'gpt-4o',
                      temperature: float = 0.0, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                      presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []
    if len(history) > 0:
        usr_content.append({"type": "text", "text": f"These are history obervations of the robot:"})
        for idx, history_image in enumerate(history):
            usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{history_image}", "detail": "high"}})
        usr_content.append({"type": "text", "text": f"The following are the results of the sampled actions you need to choose from."})
    
    letter_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    for idx, result in enumerate(results):
        usr_content.append({"type": "text", "text": f"This is the obervation of future result {letter_list[idx]}:"})
        image = result[0]
        usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # temperature=temperature,
        # max_tokens=max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content


def simple_generate_response_concatenate(results: list, system_prompt: str, motion_name_list: list = None, history = None, model: str = 'gpt-4o',
                      temperature: float = 0.0, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                      presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []
    for idx, result in enumerate(results):
        usr_content.append({"type": "text", "text": f"This is the image the multi-view observations of result {idx + 1}."})
        if motion_name_list is not None:
            usr_content.append({"type": "text", "text": f'The motion direction relative to the robot base is {motion_name_list[idx]}'})
        concatenated_image = result[0]
        usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{concatenated_image}", "detail": "high"}})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # temperature=temperature,
        # max_tokens=max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content


def simple_generate_response_seperate(results: list, system_prompt: str, history = None, model: str = 'gpt-4o',
                      temperature: float = 0.0, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                      presence_penalty: float = 0.0, stop: list = None):
    usr_content = []
    for idx, result in enumerate(results):
        usr_content.append({"type": "text", "text": f"These are the multi-view images of result {idx + 1}."})
        for image in result:
            usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # temperature=temperature,
        # max_tokens=max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content


def simple_generate_choice(descriptions: list, system_prompt: str, history = None, model: str = 'gpt-4o',
                    temperature: float = 0.0, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                    presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []
    for idx, description in enumerate(descriptions):
        usr_content.append({"type": "text", "text": f'This is the description of result {idx + 1} : {description}'})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # temperature=temperature,
        # max_tokens=max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content


def simple_generate_response_voting_concatenate(results: list, system_prompt: str, motion_name_list: list = None, history = None, model: str = 'gpt-4o',
                      temperature: float = 0.0, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                      presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []
    for idx, result in enumerate(results):
        usr_content.append({"type": "text", "text": f"This is the observations of all results from view {idx + 1}."})
        if motion_name_list is not None:
            usr_content.append({"type": "text", "text": f'The motion direction relative to the robot base is {motion_name_list[idx]}'})
        concatenated_image = result[0]
        usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{concatenated_image}", "detail": "high"}})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # temperature=temperature,
        # max_tokens=max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content


def simple_generate_response_temperature(results: list, system_prompt: str, history = None, model: str = 'gpt-4o',
                      temperature: float = 0.2, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                      presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []
    for idx, result in enumerate(results):
        usr_content.append({"type": "text", "text": f"This is the obervation of result {idx + 1}."})
        image = result[0]
        usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})
    
    payload = {
        "model" : f"{model}",
        "temperature": temperature,
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # temperature=temperature,
        # max_tokens=max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content


def simple_generate_response_concatenate_temperature(results: list, system_prompt: str, motion_name_list: list = None, history = None, model: str = 'gpt-4o',
                      temperature: float = 0.2, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                      presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []
    for idx, result in enumerate(results):
        usr_content.append({"type": "text", "text": f"This is the image the multi-view observations of result {idx + 1}."})
        if motion_name_list is not None:
            usr_content.append({"type": "text", "text": f'The motion direction relative to the robot base is {motion_name_list[idx]}'})
        concatenated_image = result[0]
        usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{concatenated_image}", "detail": "high"}})
    
    payload = {
        "model" : f"{model}",
        "temperature": temperature,
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # temperature=temperature,
        # max_tokens=max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content


def simple_generate_response_current(results: list, system_prompt: str, history = None, model: str = 'gpt-4o',
                      temperature: float = 0.0, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                      presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []

    usr_content.append({"type": "text", "text": f"This is the obervation of the current state:"})
    image = results[0][0]
    usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})

    for idx, result in enumerate(results[1:]):
        usr_content.append({"type": "text", "text": f"This is the obervation of result {idx + 1}."})
        image = result[0]
        usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # temperature=temperature,
        # max_tokens=max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content


def simple_generate_response_concatenate_current(results: list, system_prompt: str, motion_name_list: list = None, history = None, model: str = 'gpt-4o',
                      temperature: float = 0.0, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                      presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []

    usr_content.append({"type": "text", "text": f"This is the image containing multi-view obervation of the current state:"})
    image = results[0][0]
    usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})

    for idx, result in enumerate(results[1:]):
        usr_content.append({"type": "text", "text": f"This is the image containing multi-view observations of result {idx + 1}."})
        if motion_name_list is not None:
            usr_content.append({"type": "text", "text": f'The motion direction relative to the robot base is {motion_name_list[idx]}'})
        concatenated_image = result[0]
        usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{concatenated_image}", "detail": "high"}})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # "temperature": temperature,
        # "max_tokens": max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content

def simple_generate_response_incontext(results: list, system_prompt: str, history = None, examples = None, model: str = 'gpt-4o',
                      temperature: float = 0.0, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                      presence_penalty: float = 0.0, stop: list = None):
    usr_content = []
    usr_content.append({"type": "text", "text": f"First we will show you some examples."})
    for idx, example in enumerate(examples):
        usr_content.append({"type": "text", "text": f"This is one example:"})
        usr_content.append({"type": "text", "text": f'The goal of robot in this example is: {example[0]}.'})
        for action_id, example_image in enumerate(example[1]):
            usr_content.append({"type": "text", "text": f"This is the obervation of result {action_id + 1} in this example:"})
            usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{example_image}", "detail": "high"}})
        usr_content.append({"type": "text", "text": f'This is the answer of this example: {example[2]}'})

    usr_content.append({"type": "text", "text": f"Below is the real observation of results you need to handle."})

    for idx, result in enumerate(results):
        usr_content.append({"type": "text", "text": f"This is the obervation of result {idx + 1}:"})
        image = result[0]
        usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # temperature=temperature,
        # "max_tokens": max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content


def simple_generate_response_current_incontext(results: list, system_prompt: str, history = None, examples = None, model: str = 'gpt-4o',
                      temperature: float = 0.0, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                      presence_penalty: float = 0.0, stop: list = None):
    usr_content = []
    usr_content.append({"type": "text", "text": f"First we will show you some examples."})
    for idx, example in enumerate(examples):
        usr_content.append({"type": "text", "text": f"This is one example:"})
        usr_content.append({"type": "text", "text": f'The goal of robot in this example is: {example[0]}.'})
        usr_content.append({"type": "text", "text": f"This is the obervation of the current state in this example:"})
        image = example[1][0]
        usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})
        for action_id, example_image in enumerate(example[1][1:]):
            usr_content.append({"type": "text", "text": f"This is the obervation of result {action_id + 1} in this example:"})
            usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{example_image}", "detail": "high"}})
        usr_content.append({"type": "text", "text": f'This is the answer of this example: {example[2]}'})

    usr_content.append({"type": "text", "text": f"Below is the real observation of results you need to handle."})

    usr_content.append({"type": "text", "text": f"This is the obervation of the current state:"})
    image = results[0][0]
    usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})

    for idx, result in enumerate(results[1:]):
        usr_content.append({"type": "text", "text": f"This is the obervation of result {idx + 1}:"})
        image = result[0]
        usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # temperature=temperature,
        # "max_tokens": max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content


def simple_generate_response_concatenate_current_incontext(results: list, system_prompt: str, motion_name_list: list = None, history = None, model: str = 'gpt-4o',
                      temperature: float = 0.0, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                      presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []
    usr_content.append({"type": "text", "text": f"First we will show you some examples."})
    for idx, example in enumerate(examples):
        usr_content.append({"type": "text", "text": f"This is one example:"})
        usr_content.append({"type": "text", "text": f'The goal of robot in this example is: {example[0]}.'})
        usr_content.append({"type": "text", "text": f"This is the image containing multi-view obervation of the current state in this example:"})
        image = example[1][0]
        usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})
        for action_id, example_image in enumerate(example[1][1:]):
            usr_content.append({"type": "text", "text": f"This is the image containing multi-view observations of result {action_id + 1} in this example:"})
            usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{example_image}", "detail": "high"}})
        usr_content.append({"type": "text", "text": f'This is the answer of this example: {example[2]}'})

    usr_content.append({"type": "text", "text": f"Below is the real observation of results you need to handle."})

    usr_content.append({"type": "text", "text": f"This is the image containing multi-view obervation of the current state:"})
    image = results[0][0]
    usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})

    for idx, result in enumerate(results[1:]):
        usr_content.append({"type": "text", "text": f"This is the image containing multi-view observations of result {idx + 1}."})
        if motion_name_list is not None:
            usr_content.append({"type": "text", "text": f'The motion direction relative to the robot base is {motion_name_list[idx]}'})
        concatenated_image = result[0]
        usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{concatenated_image}", "detail": "high"}})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # "temperature": temperature,
        # "max_tokens": max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content


def simple_select_view(images: list, system_prompt: str, history = None, examples = None, model: str = 'gpt-4o',
                      temperature: float = 0.7, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                      presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []
    if examples is not None:
        usr_content.append({"type": "text", "text": f"First we will show you some examples."})
        for idx, example in enumerate(examples):
            usr_content.append({"type": "text", "text": f"This is one example:"})
            usr_content.append({"type": "text", "text": f'The goal of robot in this example is: {example[0]}.'})
            for view_id, example_image in enumerate(example[1]):
                usr_content.append({"type": "text", "text": f"This is the image obervation of the current state from view {view_id + 1} in this example:"})
                usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{example_image}", "detail": "high"}})
            usr_content.append({"type": "text", "text": f'This is the answer of this example: {example[2]}'})

        usr_content.append({"type": "text", "text": f"Below are the real observations you need to handle."})

    for idx, image in enumerate(images):
        usr_content.append({"type": "text", "text": f"This is the image obervation of the current state from view {idx + 1}:"})
        concatenated_image = image[0]
        usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{concatenated_image}", "detail": "high"}})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # "temperature": temperature,
        # "max_tokens": max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content


def select_stage(images: list, system_prompt: str, grasping=None, history = None, examples = None, model: str = 'gpt-4o',
                      temperature: float = 0.7, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                      presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []
    if examples is not None:
        usr_content.append({"type": "text", "text": f"First we will show you some examples."})
        for idx, example in enumerate(examples):
            usr_content.append({"type": "text", "text": f"This is one example:"})
            usr_content.append({"type": "text", "text": f'The goal of robot in this example is: {example[0]}.'})
            for view_id, example_image in enumerate(example[1]):
                usr_content.append({"type": "text", "text": f"This is the image obervation of the current state from view {view_id + 1} in this example:"})
                usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{example_image}", "detail": "high"}})
            usr_content.append({"type": "text", "text": f'This is the answer of this example: {example[2]}'})

        usr_content.append({"type": "text", "text": f"Below are the real observations you need to handle."})

    usr_content.append({"type": "text", "text": f"These are the image obervations of the current state from different views:"})
    if grasping is not None:
        if grasping:
            usr_content.append({"type": "text", "text": f"The gripper is grasping something now."})
        else:
            usr_content.append({"type": "text", "text": f"The gripper is not grasping anything now."})

    for idx, image in enumerate(images):
        concatenated_image = image[0]
        usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{concatenated_image}", "detail": "high"}})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # "temperature": temperature,
        # "max_tokens": max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content


def generate_success(images: list, system_prompt: str, grasping=None, history = None, examples = None, model: str = 'gpt-4o',
                      temperature: float = 0.7, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                      presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []
    if examples is not None:
        usr_content.append({"type": "text", "text": f"First we will show you some examples."})
        for idx, example in enumerate(examples):
            usr_content.append({"type": "text", "text": f"This is one example:"})
            usr_content.append({"type": "text", "text": f'The goal of robot in this example is: {example[0]}.'})
            for view_id, example_image in enumerate(example[1]):
                usr_content.append({"type": "text", "text": f"This is the image obervation of the current state from view {view_id + 1} in this example:"})
                usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{example_image}", "detail": "high"}})
            usr_content.append({"type": "text", "text": f'This is the answer of this example: {example[2]}'})

        usr_content.append({"type": "text", "text": f"Below are the real observations you need to handle."})

    usr_content.append({"type": "text", "text": f"These are the image obervations of the current state from different views:"})
    if grasping is not None:
        if grasping:
            usr_content.append({"type": "text", "text": f"The gripper is grasping something now."})
        else:
            usr_content.append({"type": "text", "text": f"The gripper is not grasping anything now."})

    for idx, image in enumerate(images):
        concatenated_image = image[0]
        usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{concatenated_image}", "detail": "high"}})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # "temperature": temperature,
        # "max_tokens": max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content


def generate_change(images: list, system_prompt: str, model: str = 'gpt-4o',
                    temperature: float = 0.7, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                    presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []
    for idx, image in enumerate(images):
        usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # temperature=temperature,
        # max_tokens=max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content


def generate_subgoals(image, system_prompt: str, model: str = 'gpt-4o',
                    temperature: float = 0.7, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                    presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []
    if image is not None:
        if isinstance(image, list):
            usr_content.append({"type": "text", "text": f"These are the image obervations of the initial state:"})
            for img in image:
                usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}", "detail": "high"}})
        else:
            usr_content.append({"type": "text", "text": f"This is the image obervation of the initial state:"})
            usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})

    usr_content.append({"type": "text", "text": f"Please break down the goal into sub-goals for robot."})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # temperature=temperature,
        # max_tokens=max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content


def generate_structured_subgoals(image, system_prompt: str, instruction: str, objects: list, examples: list, model: str = 'gpt-4o',
                    temperature: float = 0.7, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                    presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []

    object_names = ""
    for obj in objects:
        object_names += f"{obj}, "

    for example_id, example in enumerate(examples):
        encoded_image = example[0]
        text = example[1]
        # print(f"example_id: {example_id}, text: {text}")
        usr_content.append({"type": "text", "text": f"This is example {example_id + 1}:"})
        usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}", "detail": "high"}})
        usr_content.append({"type": "text", "text": text})
    
    usr_content.append({"type": "text", "text": f"Below are the real observation and instruction you need to handle."})

    usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})
    usr_content.append({"type": "text", "text": f"instruction is: {instruction}, objects in the scene: {object_names}"})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # temperature=temperature,
        # max_tokens=max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content


def generate_grasp(image, system_prompt: str, model: str = 'gpt-4o',
                    temperature: float = 0.7, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                    presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []
    usr_content.append({"type": "text", "text": f"These are the image obervations after the grasping the object:"})
    for img in image:
        usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}", "detail": "high"}})
    # usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})

    usr_content.append({"type": "text", "text": f"Please tell whether grasping this object align with the goal of the robot."})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # temperature=temperature,
        # max_tokens=max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content


def generate_release(image, system_prompt: str, model: str = 'gpt-4o',
                    temperature: float = 0.7, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                    presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []

    # careful! not compatible with single image
    usr_content.append({"type": "text", "text": f"These are the image obervations after the releasing the object:"})
    for idx, img in enumerate(image):
        concatenated_image = img[0]
        usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{concatenated_image}", "detail": "high"}})
    # image = image[0]
    # usr_content.append({"type": "text", "text": f"This is the image obervation after the releasing the object:"})
    # usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})

    usr_content.append({"type": "text", "text": f"Please tell whether releasing this object align with the goal of the robot."})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # temperature=temperature,
        # max_tokens=max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content


def generate_toss(image, prev_image, system_prompt: str, model: str = 'gpt-4o',
                    temperature: float = 0.7, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                    presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []

    # careful! not compatible with single image
    usr_content.append({"type": "text", "text": f"This is the image obervations before the tossing the object:"})
    usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{prev_image}", "detail": "high"}})

    usr_content.append({"type": "text", "text": f"This is the image obervations after the tossing the object:"})
    usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})

    usr_content.append({"type": "text", "text": f"Please tell whether tossing this object align with the goal of the robot."})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # temperature=temperature,
        # max_tokens=max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content


def generate_close_gripper(system_prompt: str, model: str = 'gpt-4o',
                    temperature: float = 0.7, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                    presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []
    usr_content.append({"type": "text", "text": f'Do you think the robot should keep the gripper closed during the whole process?'})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    content = response['choices'][0]['message']['content']

    return content


def test_number_of_images(number_of_images: int, model: str = 'gpt-4o',
                    temperature: float = 0.7, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                    presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []
    for i in range(number_of_images):
        with open(f'{i}.png', 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}", "detail": "high"}})
    
    usr_content.append({"type": "text", "text": f'How many images can you see? In which image the flying gripper is closer to the green cucumber?'})

    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": "You are a helpful agent"},
            {"role": "user", "content": usr_content},
        ],
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    content = response['choices'][0]['message']['content']

    return content


def prompt_helper(group_id, queue, prompt, system_prompt, grasping=False):
    try_time = 0
    change = None
    answer = None
    while change is None and try_time < 5:
        try_time += 1
        try:
            content = simple_generate_response(prompt, system_prompt, grasping=grasping)
            answer = get_answer(content)
            change = True

        except Exception as e:
            print('catched', e)
            pass
    
    if change is None:
        print('Warning: failed to match format')
        answer = 1

    queue.put((group_id, answer, content))


def prompt_release_helper(release_id, queue, prompt, system_prompt):
    try_time = 0
    change = None
    release = None
    while change is None and try_time < 5:
        try_time += 1
        try:
            content = generate_release(prompt, system_prompt)
            
            release = get_release(content)
            change = True

        except Exception as e:
            print('catched', e)
            pass
    
    if change is None:
        print('Warning: failed to match format')
        release = False

    queue.put((release_id, release, content))


def generate_segment_names(system_prompt: str, image, instruction: str, model: str = 'gpt-4o',
                      temperature: float = 0.0, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                      presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []

    usr_content.append({"type": "text", "text": f"The instruction of the task is: {instruction}"})
    # image = results[0][0]
    usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})

    # for idx, result in enumerate(results[1:]):
    #     usr_content.append({"type": "text", "text": f"This is the image containing multi-view observations of result {idx + 1}."})
    #     if motion_name_list is not None:
    #         usr_content.append({"type": "text", "text": f'The motion direction relative to the robot base is {motion_name_list[idx]}'})
    #     concatenated_image = result[0]
    #     usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{concatenated_image}", "detail": "high"}})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # "temperature": temperature,
        # "max_tokens": max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content


def generate_reward_function(system_prompt: str, instruction: str, objects: list, model: str = 'gpt-4o',
                      temperature: float = 0.0, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                      presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []
    object_names = ""
    for obj in objects:
        object_names += f'"{obj}", '

    usr_content.append({"type": "text", "text": f"instruction: {instruction} objects: {object_names}"})

    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # "temperature": temperature,
        # "max_tokens": max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content


# context = test_number_of_images(number_of_images=4)
# print(context)