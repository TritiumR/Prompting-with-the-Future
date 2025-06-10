import os
import base64
import matplotlib.pyplot as plt


prompt_path = 'prompts'
example_path = 'examples'

def get_prompt(args):
    prompt_name = 'evaluate_system_prompt'

    prompt_name = prompt_name + '.txt'

    print('using prompt: ', prompt_name)

    with open(f'{prompt_path}/{prompt_name}', 'r') as f:
        system_prompt = f.read()

    system_prompt = system_prompt.replace('<instruction>', args.instruction)

    return system_prompt


# select view prompt
def get_view_prompt(args):
    view_prompt_name = 'view_system_prompt'

    view_prompt_name = view_prompt_name + '.txt'

    print('using view prompt: ', view_prompt_name)

    with open(f'{prompt_path}/{view_prompt_name}', 'r') as f:
        select_prompt = f.read()
    
    select_prompt = select_prompt.replace('<instruction>', args.instruction)

    
    return select_prompt


# select stage prompt
def get_stage_prompt(args):
    prompt_name = 'stage_system_prompt'

    prompt_name = prompt_name + '.txt'

    print('using stage prompt: ', prompt_name)

    with open(f'{prompt_path}/{prompt_name}', 'r') as f:
        prompt = f.read()
    
    prompt = prompt.replace('<instruction>', args.instruction)

    return prompt


def get_subgoal_prompt(args):
    subgoal_system_prompt_name = 'subgoal_system_prompt'

    subgoal_system_prompt_name = subgoal_system_prompt_name + '.txt'

    print('using subgoal prompt: ', subgoal_system_prompt_name)

    with open(f'{prompt_path}/{subgoal_system_prompt_name}', 'r') as f:
        subgoal_prompt = f.read()

    subgoal_prompt = subgoal_prompt.replace('<instruction>', args.instruction)

    return subgoal_prompt


def get_rotate_prompt(args):
    rotate_system_prompt_name = 'rotate_system_prompt'

    rotate_system_prompt_name = rotate_system_prompt_name + '.txt'

    print('using rotate prompt: ', rotate_system_prompt_name)

    with open(f'{prompt_path}/{rotate_system_prompt_name}', 'r') as f:
        rotate_prompt = f.read()

    rotate_prompt = rotate_prompt.replace('<instruction>', args.instruction)

    return rotate_prompt


def get_grasp_prompt(args):
    system_prompt_name = 'grasp_system_prompt'

    system_prompt_name = system_prompt_name + '.txt'

    print('using grasp prompt: ', system_prompt_name)

    with open(f'{prompt_path}/{system_prompt_name}', 'r') as f:
        prompt = f.read()

    prompt = prompt.replace('<instruction>', args.instruction)

    return prompt


def get_release_prompt(args):
    system_prompt_name = 'release_system_prompt'

    system_prompt_name = system_prompt_name + '.txt'

    print('using release prompt: ', system_prompt_name)

    with open(f'{prompt_path}/{system_prompt_name}', 'r') as f:
        prompt = f.read()

    prompt = prompt.replace('<instruction>', args.instruction)

    return prompt


def get_close_gripper_prompt(args):
    system_prompt_name = 'close_gripper_system_prompt'

    system_prompt_name = system_prompt_name + '.txt'

    print('using close gripper prompt: ', system_prompt_name)

    with open(f'{prompt_path}/{system_prompt_name}', 'r') as f:
        prompt = f.read()

    prompt = prompt.replace('<instruction>', args.instruction)

    return prompt


def get_success_prompt(args):
    system_prompt_name = 'success_system_prompt'

    system_prompt_name = system_prompt_name + '.txt'

    print('using success prompt: ', system_prompt_name)

    with open(f'{prompt_path}/{system_prompt_name}', 'r') as f:
        prompt = f.read()

    prompt = prompt.replace('<instruction>', args.instruction)

    return prompt

