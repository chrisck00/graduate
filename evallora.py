# Import necessary modules
import torch
import torch.nn as nn
import sys
# Import get_loaders function from data module within the same directory
from data import get_loaders
import fnmatch
from pdb import set_trace as st

# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(model, tokenizer, device=torch.device("cuda:0"), dataset="wikitext2"):

    # Print status
    print(f"evaluating on {dataset}")

    # Get the test loader
    _, testloader = get_loaders(
        dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer 
    )

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl = eval_ppl_dataset(model, testloader, 1, device)
    return ppl 

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_dataset(model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # st()
        
        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        # inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to("cuda:1")
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)


        # print ("nlls",nlls)
        sys.stdout.flush()

    
    print ('begin calcualte ppl')
    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    # torch.cuda.empty_cache()

    return ppl.item()




def eval_zero_shot(model_name, task_list=["qqp","rte","mnli","mrpc","cola", "qnli", "stsb"], 
        num_fewshot=0, use_accelerate=True, add_special_tokens=False):
    from lm_eval import tasks, evaluator
    from lm_eval.utils import make_table
    import csv
    from transformers import logging as hf_logging

    hf_logging.set_verbosity_error()
    # from lm_eval.tasks import TaskManager
    # tm = TaskManager() 
    # def pattern_match(patterns, source_list):
    #     task_names = set()
    #     for pattern in patterns:
    #         for matching in fnmatch.filter(source_list, pattern):
    #             task_names.add(matching)
    #     return list(task_names)
    # task_names = pattern_match(task_list, tm.all_tasks)
    
    model_args = f"pretrained={model_name},cache_dir={model_name}"
    # model_name="meta-llama/Llama-2-7b-hf"
    # peft_path="Styxxxx/llama2_7b_lora-hellaswag"
    # model_args = f"pretrained={model_name},peft={peft_path}"
    # if use_accelerate:
    #     model_args = f"pretrained={model_name},use_accelerate=True,device_map_option=\"auto\""
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=task_list,
        num_fewshot=num_fewshot,
        batch_size=None,
        max_batch_size=None,
        device="cuda",
        # limit=limit,
        check_integrity=False,
        write_out=False,
    )
    print("********************************")
    print("zero_shot evaluation results")
    print(make_table(results))
    # st()
    
    table_str = make_table(results)
    lines = table_str.strip().split("\n")

    # 1. header
    header_line = lines[0]
    raw_headers = [h.strip() for h in header_line.strip('|').split('|')]

    # 불필요한 빈 컬럼 제거 (예: '', '↑', '±' 같은 것들)
    headers = [h for h in raw_headers if h not in {"", "↑", "±"}]

    # 2. 데이터 라인
    data_lines = lines[2:]

    parsed_rows = []
    for line in data_lines:
        if line.strip().startswith("|"):
            raw_values = [v.strip() for v in line.strip('|').split('|')]
            values = [v for v in raw_values if v not in {"", "↑", "±"}]
            parsed_rows.append(values)

    # 3. 저장
    with open("anli_results.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(parsed_rows)

    print("✅ CSV 저장 완료: results_from_make_table.csv")
    return results