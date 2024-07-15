import json
import os
import argparse
import logging
from utils import get_gpt_response_openai



logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)



def check_cleansing(eval_system):
    all_folders = os.listdir('./data/')
    for folder in all_folders:
        if os.path.isdir(folder) and not folder.startswith('__') and not folder.startswith('.') and not folder.startswith('data'):
            jsonlines = open(f'./data/{folder}/{folder}_qa.jsonl', 'r').readlines()
            system_answers = []
            cur_qa_idx = 1
            cur_content = ""
            if eval_system == 'ernie4':
                with open(f'./data/{folder}/{eval_system}_results.txt', 'r') as f:
                    for line in f.readlines():
                        line = line.strip()
                        if line != "":
                            system_answers.append(line)
            else:
                with open(f'./data/{folder}/{eval_system}_results.txt', 'r') as f:
                    for line in f.readlines():
                        line = line.strip()
                        if not line.startswith('1.'): pass
                        if line != "":
                            if line.startswith(f'{cur_qa_idx}.'):
                                cur_content = line
                            elif line.startswith(f'{cur_qa_idx+1}.'):
                                system_answers.append(cur_content)
                                cur_content = line
                                cur_qa_idx += 1
                            else:
                                cur_content += line
                system_answers.append(cur_content)

            if len(system_answers) != len(jsonlines): 
                raise Exception(f'Check the {folder}-folder')
    return 

def align_eval_input(eval_system):
    if os.path.exists(f'./{eval_system}_eval_input.jsonl'): return
    all_folders = os.listdir('./data/')
    for folder in all_folders:
        if os.path.isdir(folder) and not folder.startswith('__') and not folder.startswith('.') and not folder.startswith('data'):
            system_answers = []
            with open(f'./data/{folder}/{eval_system}_results.txt', 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line != "":
                        system_answers.append(line.strip())
            
            jsonlines = open(f'./data/{folder}/{folder}_qa.jsonl', 'r').readlines()
            new_dict_list = []
            for i, jsonline in enumerate(jsonlines):
                system_ans = system_answers[i]
                system_ans = system_ans.lstrip(f'{i+1}.').strip()
                jsonline = json.loads(jsonline)
                jsonline['sys_ans'] = system_ans
                jsonline['file'] = folder
                new_dict_list.append(jsonline)
            
            with open(f'./{eval_system}_eval_input.jsonl', 'a') as f:
                for json_dict in new_dict_list:
                    f.write(json.dumps(json_dict) + '\n')

    return


def evaluate(eval_system, resume_id=0):
    # read evaluation prompt
    eval_prompt_dir = './evaluation_prompt.txt'
    eval_prompt = open(eval_prompt_dir).read()
    system_content = 'You are a helpful evaluator.'

    eval_inp_dir = f'./{eval_system}_eval_input.jsonl'
    eval_out_dir = f'./{eval_system}_eval_output.jsonl'

    with open(eval_inp_dir, 'r') as f:
        json_dict_list = [json.loads(line) for line in f.readlines()]

    for i, json_dict in enumerate(json_dict_list):
        if i < resume_id:
            continue
        question, sys_ans, ref_ans, ref_text = json_dict['question'], json_dict['sys_ans'], json_dict['answer'], json_dict['evidence']
        cur_prompt = eval_prompt.replace('{{question}}', question).replace('{{sys_ans}}', sys_ans).replace('{{ref_ans}}', ref_ans).replace('{{ref_text}}', ref_text)
        response = get_gpt_response_openai(cur_prompt, system_content=system_content)
        json_dict['eval'] = response

        with open(eval_out_dir, 'a') as f:
            f.write(json.dumps(json_dict) + '\n')
        print(f"-Finish {i}-th qa")
    return



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", type=str, default="", choices=['gpt-4o','gpt4', 'gpt4_pl', 'gpt-4o_pl', 'gpt3.5', 'phi3-medium','commandr-35b','internlm2-20b', 'internlm2-7b', 'chatglm3-6b','gpt3.5','llama3-8b','llama3-70b','yi1.5-9b', 'yi1.5-34b','mixtral-8x7b','mistral-7b','gemma-7b', 'llama2-13b', 'kimi', 'claude3','glm4', 'qwen2.5', 'ernie4'], help="The name of evaluated system.")
    parser.add_argument("--resume_id", type=int, default=0,
                        help="From which folder to begin evaluation.")
    
    
    args = parser.parse_args()

    eval_system = args.system
    resume_id = args.resume_id

    if eval_system in ['gpt-4o','gpt4', 'gpt4_pl', 'gpt-4o_pl', 'gpt3.5', 'kimi', 'claude3','glm4', 'qwen2.5', 'ernie4']:
        check_cleansing(eval_system)
        align_eval_input(eval_system)
    evaluate(eval_system, resume_id=resume_id)



if __name__ == "__main__":
    main()




