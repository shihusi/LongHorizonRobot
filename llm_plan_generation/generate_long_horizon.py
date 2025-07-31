from openai import OpenAI
import os
import time
import argparse
import httpx
import json

def round_floats_in_list(input_list):
    return [round(x, 3) if isinstance(x, float) else x for x in input_list]
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

CHAT_GPT_VERSION = "gpt-4-turbo-preview"

def call_gpt4_planner(planner_prompt, env_description):
    print("\n>>>> begin planner!")
    response = client.chat.completions.create(
        model=CHAT_GPT_VERSION, 
        messages=[
            {"role": "system", "content": "You should strictly follow the user's requirements, and think step-by-step logically."},
            {"role": "user", "content": planner_prompt},
            {"role": "assistant", "content": 'Yes'},
            {"role": "user", "content": env_description}
        ],
        n=1,
        temperature=0.2
    )
    time.sleep(1.0)
    print("\n>> total tokens=", response.usage.total_tokens)
    abstract_plan = response.choices[0].message.content
    print("\n>> abstract_plan=", abstract_plan)
    return abstract_plan

def call_gpt4_calculator(calculator_prompt, abstract_plan, env_description):
    print("\n>>>> begin calculator!")
    response = client.chat.completions.create(
        model=CHAT_GPT_VERSION, 
        messages=[
            {"role": "system", "content": "You should strictly follow the user's requirements, and think step-by-step logically."},
            {"role": "user", "content": calculator_prompt},
            {"role": "assistant", "content": 'Yes'},
            {"role": "user", "content": f'''The environment description is {env_description}. The abstract plan is {abstract_plan}.'''}
        ],
        n=1,
        temperature=0.2
    )
    time.sleep(1.0)
    print("\n>> total tokens=", response.usage.total_tokens)
    calculate_res = response.choices[0].message.content
    print("\n>> calculator_res=", calculate_res)
    return calculate_res

def call_gpt4_coder(coder_prompt, abstract_plan, env_description, calculate_res, global_var):
    print("\n>>>> begin coder!")
    response = client.chat.completions.create(
        model=CHAT_GPT_VERSION, 
        messages=[
            {"role": "system", "content": "You should strictly follow the user's requirements, and think step-by-step logically."},
            {"role": "user", "content": coder_prompt},
            {"role": "assistant", "content": 'Yes'},
            {"role": "user", "content": f'''The environment description is {env_description}. The abstract plan is {abstract_plan}. Some calculation results are {calculate_res}. The global variables that you could use directly are {global_var}'''}
        ],
        n=1,
        temperature=0.2
    )
    time.sleep(1.0)
    print("\n>> total tokens=", response.usage.total_tokens)
    codes = response.choices[0].message.content
    print("\n>> codes=", codes)
    return codes

def call_gpt4_coder_no_cal(coder_prompt, abstract_plan, env_description, global_var):
    print("\n>>>> begin coder ablation(no calculator)!")
    response = client.chat.completions.create(
        model=CHAT_GPT_VERSION, 
        messages=[
            {"role": "system", "content": "You should strictly follow the user's requirements, and think step-by-step logically."},
            {"role": "user", "content": coder_prompt},
            {"role": "assistant", "content": 'Yes'},
            {"role": "user", "content": f'''The environment description is {env_description}. The abstract plan is {abstract_plan}. The global variables that you could use directly are {global_var}'''}
        ],
        n=1,
        temperature=0.1
    )
    time.sleep(1.0)
    print("\n>> total tokens=", response.usage.total_tokens)
    codes = response.choices[0].message.content
    print("\n>> codes=", codes)
    return codes

def call_gpt4_coder_only(coder_prompt, env_description, global_var):
    print("\n>>>> begin coder ablation(only coder)!")
    response = client.chat.completions.create(
        model=CHAT_GPT_VERSION, 
        messages=[
            {"role": "system", "content": "You should strictly follow the user's requirements, and think step-by-step logically."},
            {"role": "user", "content": coder_prompt},
            {"role": "assistant", "content": 'Yes'},
            {"role": "user", "content": f'''The environment description is {env_description}. The global variables that you could use directly are {global_var}'''}
        ],
        n=1,
        temperature=0.1
    )
    time.sleep(1.0)
    print("\n>> total tokens=", response.usage.total_tokens)
    codes = response.choices[0].message.content
    print("\n>> codes=", codes)
    return codes

def gen_main_res(planner_prompt,
                 coder_prompt,
                 calculator_prompt,
                 env_description,
                 global_var):

    abstract_plan = call_gpt4_planner(planner_prompt, env_description)
    calculate_res = call_gpt4_calculator(calculator_prompt, abstract_plan, env_description)
    codes = call_gpt4_coder(coder_prompt, abstract_plan, env_description, calculate_res, global_var)
    return codes

def gen_ablation1_no_calculator(planner_prompt,
                                coder_prompt,
                                env_description,
                                global_var):
    
    abstract_plan = call_gpt4_planner(planner_prompt, env_description)
    codes = call_gpt4_coder_no_cal(coder_prompt, abstract_plan, env_description, global_var)
    return codes

def gen_ablation2_coder_only(coder_prompt,
                             env_description,
                             global_var):
    
    codes = call_gpt4_coder_only(coder_prompt, env_description, global_var)
    return codes

def generate_multi_long_horizon(planner_prompt, coder_prompt, calculator_prompt, env_description, global_var, coder_no_calculator_prompt, coder_only_prompt,
            env_folder, args, role_folder=r"long_horizon", gen_num=10):
    
    for i in range(gen_num):
        if args.main:
            codes = gen_main_res(planner_prompt, coder_prompt, calculator_prompt, env_description, global_var)
            with open(f"./response/{env_folder}/{role_folder}/main_codes_{str(i).zfill(3)}.md", "w", encoding="utf-8") as f:
                f.write(codes)

        if args.no_calculator:
            codes = gen_ablation1_no_calculator(planner_prompt, coder_no_calculator_prompt, env_description, global_var)
            with open(f"./response/{env_folder}/ablation/no_calculator_codes_{str(i).zfill(3)}.md", "w", encoding="utf-8") as f:
                f.write(codes)
        
        if args.coder_only:
            codes = gen_ablation2_coder_only(coder_only_prompt, env_description, global_var)
            with open(f"./response/{env_folder}/ablation/coder_only_codes_{str(i).zfill(3)}.md", "w", encoding="utf-8") as f:
                f.write(codes)

def call_gpt4_replaner_new_goal(env_folder, replaner_prompt, env_description, global_var, skill, new_goal, gen_id=2, test_id=1, case_id=1):
    print("\n>>>> begin replanner, in the case where goal changes!")
    
    with open(f"./response/{env_folder}/long_horizon/execute_code/main_codes_00{gen_id}.md") as f:
        original_code = f.read()
        
    response = client.chat.completions.create(
        model=CHAT_GPT_VERSION, 
        messages=[
            {"role": "system", "content": "You should strictly follow the user's requirements, and think step-by-step logically."},
            {"role": "user", "content": replaner_prompt},
            {"role": "assistant", "content": 'Yes'},
            {"role": "user", "content": f'''The environment description is {env_description}.  The original code is {original_code}. You are interrupted after you finish executing the  {skill} skill in the plan. This interruption is because the human changes the ultimate goal, and your new goal is {new_goal}. The global variables that you could use directly are {global_var}.'''}
        ],
        n=1,
        temperature=0.2
    )
    
    time.sleep(1.0)
    print("\n>> total tokens=", response.usage.total_tokens)
    codes = response.choices[0].message.content
    print("\n>> codes=", codes)

    os.makedirs(f"./response/{env_folder}/long_horizon/replan_dev", exist_ok=True)
    with open(f"./response/{env_folder}/long_horizon/replan_dev/replan_codes_newgoal_case{case_id}_gen{gen_id}_test{test_id}.md", "w", encoding="utf-8") as f:
        f.write(codes)

    return codes

def call_gpt4_replaner_failure(env_folder, replaner_prompt, env_description, global_var, skill, failure_info, gen_id=2, test_id=1, case_id=1):
    print("\n>>>> begin replanner, in the case when failed at certain skill!")
    
    with open(f"./response/{env_folder}/long_horizon/execute_code/main_codes_00{gen_id}.md") as f:
        original_code = f.read()
        
    response = client.chat.completions.create(
        model=CHAT_GPT_VERSION, 
        messages=[
            {"role": "system", "content": "You should strictly follow the user's requirements, and think step-by-step logically."},
            {"role": "user", "content": replaner_prompt},
            {"role": "assistant", "content": 'Yes'},
            {"role": "user", "content": f'''The environment description is {env_description}.  The original code is {original_code}. You are interrupted after you finish executing the  {skill} skill in the plan. This interruption is because you failed at this skill, and your current state is: {failure_info}. The global variables that you could use directly are {global_var}.'''}
        ],
        n=1,
        temperature=0.2
    )
    
    time.sleep(1.0)
    print("\n>> total tokens=", response.usage.total_tokens)
    codes = response.choices[0].message.content
    print("\n>> codes=", codes)

    os.makedirs(f"./response/{env_folder}/long_horizon/replan_dev", exist_ok=True)
    with open(f"./response/{env_folder}/long_horizon/replan_dev/replan_codes_failure_case{case_id}_gen{gen_id}_test{test_id}.md", "w", encoding="utf-8") as f:
        f.write(codes)

    return codes

def call_gpt4_replaner(env_folder, replaner_prompt, env_description, global_var, gen_id=2, ac_id = 2, test_id=1):
    print("\n>>>> begin replanner!")

    with open(f"./prompts/envs/{env_folder}/random_data_{env_folder}.json", "r", encoding="utf-8") as f:
        error_data = json.load(f)
    
    with open(f"./response/{env_folder}/long_horizon/execute_code/main_codes_00{gen_id}.md") as f:
        original_code = f.read()

    num2idx_dict = {
        1: "first",
        2: "second",
    }
    

    fail_des_button = f'''
        During your tring to touch the button, The most relative position of the button to your hand is a variable button_to_hand[button_to_hand[0], button_to_hand[1], button_to_hand[2]], 
        you can use get_fail_info('button_to_hand') to get it, you can use the position to adjust your hand_touch_position's target.
'''
    fail_des_manipulate = f'''
        During your tring to push the object to target position, The most relative position of the target position to the object is a variable target_to_object[target_to_object[0], target_to_object[1], target_to_object[2]], 
        you can use get_fail_info('target_to_object') to get it, you can use the position to adjust your push_to_position's target.
'''


    response = client.chat.completions.create(
        model=CHAT_GPT_VERSION, 
        messages=[
            {"role": "system", "content": "You should strictly follow the user's requirements, and think step-by-step logically."},
            {"role": "user", "content": replaner_prompt},
            {"role": "assistant", "content": 'Yes'},
            # {"role": "user", "content": f'''The environment description is {env_description}.  The original code is {original_code}. You are fail at your {num2idx_dict[ac_id]} hand_touch_position. The fail_des is {fail_des_button} The global variables that you could use directly are {global_var}'''}
            {"role": "user", "content": f'''The environment description is {env_description}.  The original code is {original_code}. You are fail at your {num2idx_dict[ac_id-1]} push_to_position. The fail_des is {fail_des_manipulate} The global variables that you could use directly are {global_var}'''}
        ],
        n=1,
        temperature=0.2
    )
    time.sleep(1.0)
    print("\n>> total tokens=", response.usage.total_tokens)
    codes = response.choices[0].message.content
    print("\n>> codes=", codes)

    with open(f"./response/{env_folder}/long_horizon/replan/replan_codes_{gen_id}_ac{ac_id}_test{test_id}.md", "w", encoding="utf-8") as f:
        f.write(codes)

    return codes


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="The environment name.")
    parser.add_argument("--main", default=False, action="store_true", help="Whether to generate main results.")
    parser.add_argument("--no-calculator", default=False, action="store_true", help="Whether to run ablation 1 (planner + coder, no calculator).")
    parser.add_argument("--coder-only", default=False, action="store_true", help="Whether to run ablation 2 (coder only).")
    args = parser.parse_args()
    

    role_folder = r"long_horizon"
    env_folder = args.env 
    use_ablation = True

    # read in all prompts
    with open(f"./prompts/role/{role_folder}/planner_prompt.md", "r", encoding="utf-8") as f:
        planner_prompt = f.read()
    with open(f"./prompts/role/{role_folder}/coder_prompt.md", "r", encoding="utf-8") as f:
        coder_prompt = f.read()
    with open(f"./prompts/role/{role_folder}/calculator_prompt.md", "r", encoding="utf-8") as f:
        calculator_prompt = f.read()

    with open(f"./prompts/role/{role_folder}/replaner_prompt.md", "r", encoding="utf-8") as f:
        replaner_prompt = f.read()
    if use_ablation:
        with open(f"./prompts/role/ablation/coder_no_calculator_prompt.md", "r", encoding="utf-8") as f:
            coder_no_calculator_prompt = f.read()
        with open(f"./prompts/role/ablation/coder_only_prompt.md", "r", encoding="utf-8") as f:
            coder_only_prompt = f.read()

    # read in environment information
    with open(f"./prompts/envs/{env_folder}/env_description.md", "r", encoding="utf-8") as f:
        env_description = f.read()
    with open(f"./prompts/envs/{env_folder}/global_var.md", "r", encoding="utf-8") as f:
        global_var = f.read()

    os.makedirs(f"./response/{env_folder}/{role_folder}", exist_ok=True)
    os.makedirs(f"./response/{env_folder}/ablation", exist_ok=True)

    
    if args.main:
        codes = gen_main_res(planner_prompt, coder_prompt, calculator_prompt, env_description, global_var)
        with open(f"./response/{env_folder}/{role_folder}/main_codes.md", "w", encoding="utf-8") as f:
            f.write(codes)

    if args.no_calculator:
        codes = gen_ablation1_no_calculator(planner_prompt, coder_no_calculator_prompt, env_description, global_var)
        with open(f"./response/{env_folder}/{role_folder}/no_calculator_codes.md", "w", encoding="utf-8") as f:
            f.write(codes)
    
    if args.coder_only:
        codes = gen_ablation2_coder_only(coder_only_prompt, env_description, global_var)
        with open(f"./response/{env_folder}/{role_folder}/coder_only_codes.md", "w", encoding="utf-8") as f:
            f.write(codes)
    