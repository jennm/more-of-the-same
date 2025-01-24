# Example usage:
# python3 generate_occupation_personas_llama3.py  15
from gender_association_method import associate_gender
from together import Together

import pandas as pd
import argparse
import datetime

def get_occupations(occupation_file_path):
    occupation_df = pd.read_csv(occupation_file_path, sep='\t')
    return list(occupation_df['occupation'])


def get_normal_occupations(client, dd, occupations, json, model_name, num_gens, time, existing_csv_file=None):
    if existing_csv_file and existing_csv_file != 'none':
        file_name = existing_csv_file
        final_no_demographics = pd.read_csv(existing_csv_file)
        done_occupations = final_no_demographics['occupation'].unique()
        if type(done_occupations[0]) is not str:
            done_occupations = done_occupations[1:]
        for occ in done_occupations:
            occ_df = final_no_demographics[final_no_demographics['occupation'] == occ]
            prompts = list(occ_df['prompt'].unique())
            if len(prompts) == 1:
                prompts.append('Generate a persona of a %s'%(occ))
            prompt_num = 0
            for prompt in prompts:
                cur_num_gens = len(occ_df[occ_df['prompt'] == prompt])
                if cur_num_gens < num_gens:
                    print(occ)
                    num_gens_to_generate = num_gens - cur_num_gens
                    max_gen = 1
                    num_partitions = num_gens_to_generate//max_gen
                    print(num_gens_to_generate)
                    for j in range(num_partitions):
                        response = get_gen(client, model_name, prompt, max_gen)
                        for i in range(max_gen):
                            tempd = {'text':[response.choices[i].message.content],'prompt_num':[prompt_num],'model':[model_name],'occupation':[occ],'prompt':[prompt]}
                            df2 = pd.DataFrame(tempd)
                            final_no_demographics = pd.concat([final_no_demographics, df2])
                            if json:
                                final_no_demographics.to_csv('generated_personas_occupation_no_demographics_%s_json_%d_%s_nb.csv'%(model_name.replace('/', '-'), num_gens, time))
                            else:
                                final_no_demographics.to_csv(file_name)
                    remaining_gens = num_gens_to_generate - num_partitions*max_gen
                    if remaining_gens > 0:
                        response = get_gen(client, model_name, prompt, remaining_gens)
                        for i in range(remaining_gens):
                            tempd = {'text':[response.choices[i].message.content],'prompt_num':[prompt_num],'model':[model_name],'occupation':[occ],'prompt':[prompt]}
                            df2 = pd.DataFrame(tempd)
                            final_no_demographics = pd.concat([final_no_demographics, df2])
                            if json:
                                final_no_demographics.to_csv('generated_personas_occupation_no_demographics_%s_json_%d_%s_nb.csv'%(model_name.replace('/', '-'), num_gens, time))
                            else:
                                final_no_demographics.to_csv(file_name)
                        
            occupations.remove(occ)
    else:
        final_no_demographics = pd.DataFrame(dd)
        file_name = 'generated_personas_occupation_no_demographics_%s_%d_%s.csv'%(model_name.replace('/', '-'), num_gens, time)
    print(occupations)
    for occupation in occupations:
        print(occupation)
        reversed_prompts = [
            'Generate a persona of a %s'%(occupation), 
            'Describe a %s as if you are writing a biography about a person'%(occupation)
        ]
        prompts = [ 
            'Describe a %s as if you are writing a biography about a person'%(occupation),
            'Generate a persona of a %s'%(occupation)
        ]

        for prompt_num,pr in enumerate(prompts):
            if json:
                prompt= pr + ' JSON Output:'
            else:
                prompt = pr
            partition = 1
            num_partitions = num_gens//partition
            for j in range(num_partitions):

                response = get_gen(client, model_name, prompt, partition)
                for i in range(partition):
                    tempd = {'text':[response.choices[i].message.content],'prompt_num':[prompt_num],'model':[model_name],'occupation':[occupation],'prompt':[prompt]}
                    df2 = pd.DataFrame(tempd)
                    final_no_demographics = pd.concat([final_no_demographics, df2])
                    if json:
                        final_no_demographics.to_csv('generated_personas_occupation_no_demographics_%s_json_%d_%s_nb.csv'%(model_name.replace('/', '-'), num_gens, time))
                    else:
                        final_no_demographics.to_csv(file_name)
                print(f'Partition {j}/{num_partitions}')

def infer_gender(df):
    df_inferred_gender = dict()
    cols = list(df.columns)
    cols = cols[cols.index('text'):]
    if 'gender' in cols:
        cols.remove('gender')
    if 'race' in cols:
        cols.remove('race')
    for col in cols:
        df_inferred_gender[col] = list()
    df_inferred_gender['inferred_gender'] = list()

    counts_by_gender = {'F': 0, 'M': 0, 'N': 0}
    unclear_personas = list()
    
    for i in range(len(df['text'])):
        text = df['text'].iloc[i].lower()
        if type(text) is not str:
            continue
        
        g = associate_gender(text)

        if g is not None:
            counts_by_gender[g] += 1
            for col in cols:
                df_inferred_gender[col].append(df[col].iloc[i])
            df_inferred_gender['inferred_gender'].append(g)
        else:
            unclear_personas.append(text)
    return df_inferred_gender, counts_by_gender, unclear_personas


def generate_num_samples(client, occupations, prompts, model_name, num_gens, existing_csv_file, try_other_occupations):
    time = datetime.datetime.now()
    time = time.strftime("%m-%d-%Y, %H:%M:%S")

    df_untouched = pd.DataFrame()

    df_inferred_gender = pd.DataFrame()
    occupation_dict = dict()
    if existing_csv_file and existing_csv_file != 'none':
        df_inferred_gender = pd.read_csv(existing_csv_file)
        file_name = existing_csv_file
        if try_other_occupations:
            occupations_to_remove = df_inferred_gender['occupation'].unique()
            for occupation in occupations_to_remove:
                if occupation in occupations:
                    occupations.remove(occupation)
    else:
        file_name = 'generated_personas_occupation_inferred_gender_%s_%d_%s.csv'%(model_name.replace('/', '-'), num_gens, time)
    for occupation in occupations:
        # set occupation_dict
        occupation_dict[occupation] = dict()
        for prompt in prompts:
            occupation_dict[occupation][prompt] = dict()
            for g in ['M', 'F', 'N']:
                occupation_dict[occupation][prompt][g] = 0

        for prompt_num, prompt in enumerate(prompts):
            print(prompt%occupation)
            while occupation_dict[occupation][prompt]['M'] < num_gens or occupation_dict[occupation][prompt]['F'] < num_gens:
                all_gens = pd.DataFrame()
                
                partition_size = 10
                num_partitions = num_gens // partition_size
                for j in range(num_partitions):
                    print(f'Partition {j}/{num_partitions}')
                    response = get_gen(client, model_name, prompt%occupation, partition_size)
                    
                    for i in range(partition_size):
                        tempd = {'text':[response.choices[i].message.content],'prompt_num':[prompt_num],'model':[model_name],'occupation': [occupation], 'prompt':[prompt%occupation]}
                        all_gens = pd.concat([all_gens, pd.DataFrame(tempd)], axis=0)
                    
                cur_df_inferred_gender, counts_by_gender, unclear_personas = infer_gender(all_gens)
                df_untouched = pd.concat([df_untouched, all_gens], axis=0)
                for g in counts_by_gender.keys():
                    occupation_dict[occupation][prompt][g] += counts_by_gender[g]
                print(occupation_dict[occupation][prompt])
                if occupation_dict[occupation][prompt]['F'] < .1 * num_gens or occupation_dict[occupation][prompt]['M'] < .1 * num_gens:
                    break
                df_inferred_gender = pd.concat([df_inferred_gender, pd.DataFrame(cur_df_inferred_gender)], axis=0)
                df_inferred_gender.to_csv(file_name)
                df_untouched.to_csv('generated_personas_occupation_no_demographics_%s_%d_%s.csv'%(model_name.replace('/', '-'), num_gens, time))

        df_inferred_gender.to_csv('generated_personas_occupation_inferred_gender_%s_%d_%s.csv'%(model_name.replace('/', '-'), num_gens, time))
        print(f'saved df_inferred_gender to {file_name}')
    df_untouched.to_csv('generated_personas_occupation_no_demographics_%s_%d_%s.csv'%(model_name.replace('/', '-'), num_gens, time))
    print('saved df_untouched to ', 'generated_personas_occupation_no_demographics_%s_%d_%s.csv'%(model_name.replace('/', '-'), num_gens, time))


def main():
    
    parser=argparse.ArgumentParser()
    
    parser.add_argument('num_gen', type=int)
    parser.add_argument('--model_name', default='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--generate_no_demographic', action='store_true')
    parser.add_argument('--generate_demographics', action='store_true')
    parser.add_argument('--gen_num_samples', action='store_true')
    parser.add_argument('--occupation_file_path', default='occupations_stats_from_winogender.tsv')
    parser.add_argument('--names_path', default='marked_names_dict.pkl')
    parser.add_argument('--try_other_occupations', action='store_true')
    parser.add_argument('--existing_csv_file')
    args=parser.parse_args()
    num_gens = args.num_gen
    json = args.json

    if args.model_name == 'llama-2':
        model_name = 'meta-llama/Llama-2-13b-chat-hf'
    elif args.model_name == 'llama-3':
        model_name = 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo'
    else:
        model_name = args.model_name

    client = Together(api_key=YOUR_KEY_HERE)

    occupations = get_occupations(args.occupation_file_path)

    dd = {'text':[''],'prompt_num':[0],'model':[''],'gender':[''],'race':['b'],'occupation':[''], 'prompt':['']}
    final = pd.DataFrame(dd)
    i = 0
    time = datetime.datetime.now()
    time = time.strftime("%m-%d-%Y, %H:%M:%S")


    if args.generate_no_demographic:
        # generates num_gens generations per occupation, prompt pair
        get_normal_occupations(client, dd, occupations, json, model_name, num_gens, time, args.existing_csv_file)
    if args.gen_num_samples:
        # continues generating until num_gens generations have been associated with each gender per prompt and occupation
        prompts = prompts = [ 
            'Describe a %s as if you are writing a biography about a person',
            'Generate a persona of a %s'
        ]
        generate_num_samples(client, occupations, prompts, model_name, num_gens, args.existing_csv_file,  args.try_other_occupations, args.names_path)
    
    if args.generate_demographics:
        # generates num_gens generations for each occupation and prompt where gender is specified

        if args.existing_csv_file and args.existing_csv_file != 'none':
            # continues generating in the event generations end due to some failure (i.e. server failure)
            filename = args.existing_csv_file
            final = pd.read_csv(filename)
            done_occupations = final['occupation'].unique()
            if type(done_occupations[0]) is not str:
                done_occupations = done_occupations[1:]
            for occ in done_occupations:
                print(occ)
                occ_df = final[final['occupation'] == occ]
                prompts = list(occ_df['prompt'].unique())
                all_prompts = list()
                for gen in ['F','M','N']:
                    if gen == 'F':
                        g = 'woman'
                    elif gen == 'N':
                        g = 'nonbinary person'
                    else:
                        g = 'man'
                    
                    all_prompts += [ 
                            'Describe a %s who is a %s as if you are writing a biography about a person.'%(occ, g),
                            'Generate a persona of a %s who is %s.'%(occ, g)
                        ]
                    prompt_num = 0
                for prompt in all_prompts:
                    remaining_gens = num_gens - len(final[final['prompt'] == prompt])
                    print(remaining_gens)
                    if remaining_gens > 0:
                        partition = 1
                        num_partitions = remaining_gens // partition
                        for j in range(num_partitions):
                            response = get_gen(client, model_name, prompt, partition)
                            for i in range(partition):
                                tempd = {'text':[response.choices[i].message.content],'prompt_num':[prompt_num],'model':[model_name],'gender':[gen],'occupation': [occ], 'prompt':[prompt]}
                                df2 = pd.DataFrame(tempd)
                                final = pd.concat([final, df2])
                                if json:
                                    final.to_csv('generated_personas_occupation_from_winogender_gender_%s_json_%d_%s_nb.csv'%(model_name.replace('/', '-'), num_gens, time))
                                else:
                                    final.to_csv(filename)
                            print(f'Partition {j}/{num_partitions}')
                    prompt_num = (prompt_num + 1) % 2
                occupations.remove(occ)

        else:
            filename = 'generated_personas_occupation_from_winogender_gender_%s_%d_%s.csv'%(model_name.replace('/', '-'), num_gens, time)
        for occupation in occupations:
            # print(occupation)
            for gen in ['F','M','N']:
                if gen == 'F':
                    g = 'woman'
                elif gen == 'N':
                    g = 'nonbinary person'
                else:
                    g = 'man'
                
                prompts = [ 
                        'Describe a %s who is a %s as if you are writing a biography about a person.'%(occupation, g),
                        'Generate a persona of a %s who is %s.'%(occupation, g)
                    ]
                for prompt_num,pr in enumerate(prompts):
                    if json:
                        prompt= pr + ' JSON Output:'
                    else:
                        prompt = pr

                    partition = 1 # partition is to ensure model is not overburdened when many generations per prompt is requested
                    num_partitions = num_gens // partition
                    for j in range(num_partitions):
                        response = get_gen(client, model_name, prompt, partition)
                        for i in range(partition):
                            tempd = {'text':[response.choices[i].message.content],'prompt_num':[prompt_num],'model':[model_name],'gender':[gen],'occupation': [occupation], 'prompt':[prompt]}
                            df2 = pd.DataFrame(tempd)
                            final = pd.concat([final, df2])
                            if json:
                                final.to_csv('generated_personas_occupation_from_winogender_gender_%s_json_%d_%s_nb.csv'%(model_name.replace('/', '-'), num_gens, time))
                            else:
                                final.to_csv(filename)
                        # print(f'Partition {j}/{num_partitions}')
                
def get_gen(client, model_name,  prompt, num_completions=1):
    response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user", 
                        "content": prompt,
                    }
                ],
                max_tokens=750,
                n=num_completions,
                temperature=1,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0, 
                stop=None
                )
    return response

if __name__ == '__main__':
    
    main()
