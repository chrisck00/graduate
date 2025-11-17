# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
from datasets import load_dataset,load_from_disk

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # traindata = load_from_disk("datasets/wikitext/train")
    # testdata = load_from_disk("datasets/wikitext/test")
    
    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    # traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    # valdata = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
    
    traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
    
    # traindata = load_from_disk("datasets/c4/train")
    # valdata = load_from_disk("datasets/c4/validation")
    
    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

def get_ptb(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')
    
    # traindata = load_from_disk("datasets/ptb/train")
    # testdata = load_from_disk("datasets/ptb/test")

    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_hellaswag(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    traindata = load_dataset('hellaswag', split='train')
    testdata = load_dataset('hellaswag', split='validation')  # HellaSwag uses 'validation' as test split

    # Debug: Print first item to understand structure
    print("Sample item structure:", traindata[0].keys() if len(traindata) > 0 else "Empty dataset")
    if len(traindata) > 0:
        print("Sample item:", traindata[0])
    
    # Process training data - combine context and correct ending
    train_texts = []
    for item in traindata:
        try:
            # Try different possible field names
            if 'ctx' in item:
                context = item['ctx']
            elif 'context' in item:
                context = item['context']
            else:
                # Fallback - use first available text field
                context = str(list(item.values())[0])
            
            # Handle label field
            if 'label' in item:
                label = item['label']
                # Convert string label to int if necessary
                if isinstance(label, str):
                    try:
                        label = int(label)
                    except ValueError:
                        label = 0  # Default to first choice
            else:
                label = 0  # Default to first choice
            
            # Handle endings
            if 'endings' in item and isinstance(item['endings'], list) and len(item['endings']) > label:
                correct_ending = item['endings'][label]
            elif 'choices' in item and isinstance(item['choices'], list) and len(item['choices']) > label:
                correct_ending = item['choices'][label]
            else:
                # Fallback - just use context
                correct_ending = ""
            
            full_text = context + " " + correct_ending if correct_ending else context
            train_texts.append(full_text)
            
        except Exception as e:
            print(f"Error processing item: {e}")
            print(f"Item: {item}")
            continue
    
    # Encode training dataset
    trainenc = tokenizer(" ".join(train_texts), return_tensors='pt')
    
    # Process test data - combine context and correct ending
    test_texts = []
    for item in testdata:
        try:
            # Try different possible field names
            if 'ctx' in item:
                context = item['ctx']
            elif 'context' in item:
                context = item['context']
            else:
                context = str(list(item.values())[0])
            
            # Handle label field
            if 'label' in item:
                label = item['label']
                if isinstance(label, str):
                    try:
                        label = int(label)
                    except ValueError:
                        label = 0
            else:
                label = 0
            
            # Handle endings
            if 'endings' in item and isinstance(item['endings'], list) and len(item['endings']) > label:
                correct_ending = item['endings'][label]
            elif 'choices' in item and isinstance(item['choices'], list) and len(item['choices']) > label:
                correct_ending = item['choices'][label]
            else:
                correct_ending = ""
            
            full_text = context + " " + correct_ending if correct_ending else context
            test_texts.append(full_text)
            
        except Exception as e:
            print(f"Error processing test item: {e}")
            continue
    
    testenc = tokenizer("\n\n".join(test_texts), return_tensors='pt')
    
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100  # Mask all but last token for next token prediction
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_rte(nsamples, seed, seqlen, tokenizer):
    dataset = load_dataset('super_glue', 'rte')
    traindata = dataset['train']
    valdata = dataset['validation']

    def format_example(example):
        premise = example['premise']
        hypothesis = example['hypothesis']
        label = example['label']
        answer = "yes" if label == 1 else "no"
        return f"Premise: {premise}\nHypothesis: {hypothesis}\nAnswer: {answer}"

    formatted_train = [format_example(x) for x in traindata]
    train_text = " ".join(formatted_train)
    trainenc = tokenizer(train_text, return_tensors='pt')

    formatted_val = [format_example(x) for x in valdata]
    val_text = " ".join(formatted_val)
    testenc = tokenizer(val_text, return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, testenc

def get_boolq(nsamples, seed, seqlen, tokenizer):
    from datasets import load_dataset
    dataset = load_dataset("super_glue", "boolq")
    traindata = dataset["train"]
    valdata = dataset["validation"]

    def format_example(example):
        passage = example["passage"]
        question = example["question"]
        label = example["label"]
        answer = "yes" if label else "no"
        return f"Passage: {passage}\nQuestion: {question}\nAnswer: {answer}"

    # Join training examples
    formatted_train = [format_example(x) for x in traindata]
    train_text = " ".join(formatted_train)
    trainenc = tokenizer(train_text, return_tensors="pt")

    # Join validation examples
    formatted_val = [format_example(x) for x in valdata]
    val_text = " ".join(formatted_val)
    testenc = tokenizer(val_text, return_tensors="pt")

    # Generate samples from training set
    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, testenc

def get_arc_challenge(nsamples, seed, seqlen, tokenizer):
    # Load ARC Challenge dataset
    traindata = load_dataset('ai2_arc', 'ARC-Challenge', split='train')
    valdata = load_dataset('ai2_arc', 'ARC-Challenge', split='validation')

    def format_example(example):
        question = example['question']
        choices = example['choices']['text']
        answer_idx = choices.index(example['answerKey']) if example['answerKey'] in choices else 0
        answer = choices[answer_idx]
        choices_str = "; ".join(choices)
        return f"Question: {question}\nChoices: {choices_str}\nAnswer: {answer}"

    # Join training examples
    formatted_train = [format_example(x) for x in traindata]
    train_text = " ".join(formatted_train)
    trainenc = tokenizer(train_text, return_tensors='pt')

    # Join validation examples
    formatted_val = [format_example(x) for x in valdata]
    val_text = " ".join(formatted_val)
    testenc = tokenizer(val_text, return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, testenc

def get_openbookqa(nsamples, seed, seqlen, tokenizer):
    # Load OpenBookQA dataset
    traindata = load_dataset('openbookqa', 'main', split='train')
    valdata = load_dataset('openbookqa', 'main', split='validation')

    def format_example(example):
        question = example['question_stem']
        choices = example['choices']['text']
        answer_idx = example['choices']['label'].index(example['answerKey'])
        answer = choices[answer_idx]
        choices_str = "; ".join(choices)
        return f"Question: {question}\nChoices: {choices_str}\nAnswer:"

    # Join training examples
    formatted_train = [format_example(x) for x in traindata]
    train_text = " ".join(formatted_train)
    trainenc = tokenizer(train_text, return_tensors='pt')

    # Join validation examples
    formatted_val = [format_example(x) for x in valdata]
    val_text = " ".join(formatted_val)
    testenc = tokenizer(val_text, return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, testenc

def get_coqa(nsamples, seed, seqlen, tokenizer):
    # Load CoQA dataset
    dataset = load_dataset('coqa')
    traindata = dataset['train']
    valdata = dataset['validation']

    def format_example(example):
        story = example['story']
        # CoQA는 여러 Q-A 쌍을 갖는 대화형 QA
        qa_pairs = example['questions'], example['answers']
        dialog = []
        for q, a in zip(*qa_pairs):
            # 질문과 답변을 이어 붙여 문장으로 표현
            dialog.append(f"Question: {q} Answer: {a}")
        dialog_str = " ".join(dialog)
        return f"Passage: {story} {dialog_str}"

    # Format train examples
    formatted_train = [format_example(x) for x in traindata]
    train_text = " ".join(formatted_train)
    trainenc = tokenizer(train_text, return_tensors='pt')

    # Format validation examples
    formatted_val = [format_example(x) for x in valdata]
    val_text = " ".join(formatted_val)
    testenc = tokenizer(val_text, return_tensors='pt')

    # Sample training sequences
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, testenc

def get_piqa(nsamples, seed, seqlen, tokenizer):
    # Load PIQA dataset
    dataset = load_dataset('piqa')
    traindata = dataset['train']
    valdata = dataset['validation']

    def format_example(example):
        goal = example['goal']
        solution_choices = example['sol1'], example['sol2']
        label = example['label']  # 0 또는 1
        answer = solution_choices[label]
        solutions_str = f"0: {solution_choices[0]} ; 1: {solution_choices[1]}"
        return f"Goal: {goal} Choices: {solutions_str} Answer: {answer}"

    # Format training set
    formatted_train = [format_example(x) for x in traindata]
    train_text = " ".join(formatted_train)
    trainenc = tokenizer(train_text, return_tensors='pt')

    # Format validation set
    formatted_val = [format_example(x) for x in valdata]
    val_text = " ".join(formatted_val)
    testenc = tokenizer(val_text, return_tensors='pt')

    # Sample from training sequences
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100  # 마지막 토큰만 예측 목표
        trainloader.append((inp, tar))

    return trainloader, testenc

def get_squadv2(nsamples, seed, seqlen, tokenizer):
    # Load SQuAD v2.0 dataset
    dataset = load_dataset('squad_v2')
    traindata = dataset['train']
    valdata = dataset['validation']

    def format_example(example):
        context = example['context']
        question = example['question']
        # `answers['text']`는 정답이 여러 개일 수 있음. 첫 번째 사용, 없으면 빈 문자열
        answer_list = example['answers']['text']
        answer = answer_list[0] if len(answer_list) > 0 else ""
        return f"Passage: {context} Question: {question} Answer: {answer}"

    # Format train examples
    formatted_train = [format_example(x) for x in traindata]
    train_text = " ".join(formatted_train)
    trainenc = tokenizer(train_text, return_tensors='pt')

    # Format validation examples
    formatted_val = [format_example(x) for x in valdata]
    val_text = " ".join(formatted_val)
    testenc = tokenizer(val_text, return_tensors='pt')

    # Sample training sequences
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, testenc

def get_triviaqa(nsamples, seed, seqlen, tokenizer):
    # Load TriviaQA dataset
    dataset = load_dataset('trivia_qa', 'rc')
    traindata = dataset['train']
    valdata = dataset['validation']

    def format_example(example):
        context = example.get('context', '')
        question = example.get('question', '')
        # 정답이 여러 개일 수 있으므로 첫 번째 사용
        answer_list = example.get('answer', {}).get('aliases', []) or example.get('answer', {}).get('text', [])
        answer = answer_list[0] if answer_list else ""
        return f"Passage: {context} Question: {question} Answer: {answer}"

    # Format train examples
    formatted_train = [format_example(x) for x in traindata]
    train_text = " ".join(formatted_train)
    trainenc = tokenizer(train_text, return_tensors='pt')

    # Format validation set
    formatted_val = [format_example(x) for x in valdata]
    val_text = " ".join(formatted_val)
    testenc = tokenizer(val_text, return_tensors='pt')

    # Sample training sequences
    random.seed(seed)
    trainloader = []
    seq_len_total = trainenc.input_ids.shape[1]
    for _ in range(nsamples):
        if seq_len_total <= seqlen:
            # 입력 길이가 짧으면 전체를 사용
            inp = trainenc.input_ids
        else:
            i = random.randint(0, seq_len_total - seqlen - 1)
            inp = trainenc.input_ids[:, i:i + seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100  # 마지막 토큰만 예측 대상
        trainloader.append((inp, tar))

    return trainloader, testenc

def get_drop(nsamples, seed, seqlen, tokenizer):
    # Load DROP dataset
    dataset = load_dataset('drop')
    traindata = dataset['train']
    valdata = dataset['validation']

    def format_example(example):
        passage = example['passage']
        question = example['question']
        answers = example['answers_spans']['spans']
        answer = answers[0] if answers else ""
        return f"Passage: {passage}\nQuestion: {question}\nAnswer: {answer}"

    formatted_train = [format_example(x) for x in traindata]
    train_text = " ".join(formatted_train)
    trainenc = tokenizer(train_text, return_tensors='pt')

    formatted_val = [format_example(x) for x in valdata]
    val_text = " ".join(formatted_val)
    testenc = tokenizer(val_text, return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, testenc

# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    if "ptb" in name:
        return get_ptb(nsamples, seed, seqlen, tokenizer)
    if "hellaswag" in name:
        return get_hellaswag(nsamples, seed, seqlen, tokenizer)
    if "rte" in name:
        return get_rte(nsamples, seed, seqlen, tokenizer)
    if "boolq" in name:
        return get_boolq(nsamples, seed, seqlen, tokenizer)
    if "arc_challenge" in name:
        return get_arc_challenge(nsamples, seed, seqlen, tokenizer)
    if "openbookqa" in name:
        return get_openbookqa(nsamples, seed, seqlen, tokenizer)
    if "coqa" in name:
        return get_coqa(nsamples, seed, seqlen, tokenizer)
    if "piqa" in name:
        return get_piqa(nsamples, seed, seqlen, tokenizer)
    if "squad_v2" in name:
        return get_squadv2(nsamples, seed, seqlen, tokenizer)
    if "trivia_qa" in name:
        return get_triviaqa(nsamples, seed, seqlen, tokenizer)
    if "drop" in name:
        return get_drop(nsamples, seed, seqlen, tokenizer)
    
