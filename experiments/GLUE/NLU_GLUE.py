from args import *
import os
import time
import torch
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft.tuners.fourier.layer import FourierLayer
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    FourierConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    FourierModel
)
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm

args = get_args()
print(args)

torch.manual_seed(args.seed)
task = args.task
peft_type = PeftType.FOURIER
device = "cuda"
num_labels = 2
if task == "stsb":
    num_labels = 1
peft_config = FourierConfig(task_type="SEQ_CLS", inference_mode=False, n_frequency = args.n_frequency, scale = args.scale)

def log(*pargs):
    path_log = './logs_glue/' + task + '/' + args.model_name_or_path.split("-")[1] + '/bs' + str(args.bs) + 'maxlen' + str(args.max_length) + 'f_lr' + str(args.fft_lr)+ 'h_lr' + str(args.head_lr) + \
          'num' + str(args.n_frequency) + 'scale' + str(args.scale) + 'seed' + str(args.seed) + '.txt'
    print(path_log)
     # Ensure the directory exists
    directory = os.path.dirname(path_log)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(path_log, mode = 'a+') as w:
        w.write(" ".join(["{}".format(t) for t in pargs]))
        w.write("\n")

if any(k in args.model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# datasets = load_dataset("glue", task)
# metric = load_metric("glue", task)

datasets = load_dataset("super_glue", task)
metric = load_metric("super_glue", task)

def preprocess_multirc(examples):
    concatenated_inputs = [
        question + " " + answer
        for question, answer in zip(examples["question"], examples["answer"])
    ]
    tokenized_inputs = tokenizer(
        examples["paragraph"],
        concatenated_inputs,
        truncation=True,
        max_length=args.max_length,
        padding="max_length",
    )
    tokenized_inputs["labels"] = examples["label"]  # Renaming happens here
    return tokenized_inputs




def tokenize_function(examples):
    # max_length=None => use the model max length (it's actually the default)
    if task == 'sst2' or task == 'cola':
        outputs = tokenizer(examples["sentence"], truncation=True, max_length=args.max_length)
    elif task == 'qnli':
        outputs = tokenizer(examples["question"], examples["sentence"], truncation=True, max_length=args.max_length)
    elif task == 'qqp':
        outputs = tokenizer(examples["question1"], examples["question2"], truncation=True, max_length=args.max_length)
    elif task == 'boolq':  # Updated for BoolQ
        outputs = tokenizer(examples["question"], examples["passage"], truncation=True, max_length=args.max_length)
    elif task == 'multirc':
        outputs = tokenizer(examples["question"], examples["paragraph"], truncation=True, max_length=args.max_length)
    elif task == 'cb' or task == 'rte':
        outputs = tokenizer(examples["premise"], examples["hypothesis"], truncation=True, max_length=args.max_length)
    elif task == 'wic':
        outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=args.max_length)
    elif task == 'wsc':
        outputs = tokenizer(examples["text"], truncation=True, max_length=args.max_length)
    elif task == 'record':
        outputs = tokenizer(examples["query"], examples["passage"], truncation=True, max_length=args.max_length)
                 # tokenizer(examples["query"], examples["passage"], truncation=True, max_length=args.max_length)
    elif task == 'copa':  
        # Concatenate choices for each example
        outputs = tokenizer(
            examples["premise"],
            [" ".join([c1, c2]) for c1, c2 in zip(examples["choice1"], examples["choice2"])],
            truncation=True,
            max_length=args.max_length,
        )
    else:
        outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=args.max_length)
    return outputs

if task == 'sst2' or task == 'cola':
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "sentence"],
    )
elif task == 'qnli':
    tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["idx", "question", "sentence"],
    )
elif task == 'qqp':
    tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["idx", "question1", "question2"],
    )
elif task == 'boolq':  # Mapping for BoolQ
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "question", "passage"],  # Updated columns for BoolQ
    )
elif task == 'multirc':
    tokenized_datasets = datasets.map(
        preprocess_multirc,
        batched=True,
        remove_columns=["paragraph", "question", "answer", "idx", "label"],  # Remove the correct columns
    )
elif task == 'cb' or task == 'rte':
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "premise", "hypothesis"],
    )
elif task == 'wic':
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "sentence1", "sentence2", "word"],
    )
elif task == 'wsc':
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "text", "span1_text", "span2_text"],
    )
elif task == 'record':
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "passage", "query", "answers"],
    )
elif task == 'copa':
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "premise", "choice1", "choice2", "question"],
    )
else:
    tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["idx", "sentence1", "sentence2"],
    )



tokenized_datasets = tokenized_datasets.rename_column("label", "labels")


def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")


# Instantiate dataloaders.
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=args.bs)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=args.bs
)

model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path,num_labels=num_labels,return_dict=True)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model
# for param in model.classifier.parameters():
#     param.requires_grad = False

head_param = list(map(id, model.classifier.parameters()))

others_param = filter(lambda p: id(p) not in head_param, model.parameters()) 

optimizer = AdamW([
    {"params": model.classifier.parameters(), "lr": args.head_lr},
    {"params": others_param, "lr": args.fft_lr}
],weight_decay=args.weight_decay)

# Instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.06 * (len(train_dataloader) * args.num_epochs),
    num_training_steps=(len(train_dataloader) * args.num_epochs),
)
acc_list = []
model.to(device)
for epoch in range(args.num_epochs):

    # Initialize lists to hold delta_w statistics for the whole epoch
    epoch_means_before, epoch_vars_before = [], []
    epoch_means_after, epoch_vars_after = [], []
    
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

       # Collect mean and variance from Fourier layers
        for module in model.modules():
            if isinstance(module, FourierLayer):  # Only collect from layers inheriting FourierLayer
                epoch_means_before.append(module.delta_w_stats["mean_before"])
                epoch_vars_before.append(module.delta_w_stats["var_before"])
                epoch_means_after.append(module.delta_w_stats["mean_after"])
                epoch_vars_after.append(module.delta_w_stats["var_after"])

    # Calculate average mean and variance for the epoch
    if epoch_means_before:
        avg_mean_before = np.mean(epoch_means_before)
        avg_var_before = np.mean(epoch_vars_before)
        avg_mean_after = np.mean(epoch_means_after)
        avg_var_after = np.mean(epoch_vars_after)

        # Print average delta_w statistics for the epoch
        print(f"Epoch {epoch} | Before Scaling -> Mean: {avg_mean_before:.2e}, Variance: {avg_var_before:.2e}")
        print(f"Epoch {epoch} | After Scaling -> Mean: {avg_mean_after:.2e}, Variance: {avg_var_after:.2e}")



    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        # batch.to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        if task == "stsb":
            predictions = outputs.logits
        else:
            predictions = outputs.logits.argmax(dim=-1)
        # predictions, references = predictions, batch["labels"]
        predictions = outputs.logits.argmax(dim=-1).cpu().numpy()
        references = batch["labels"].cpu().numpy()  # Convert to NumPy
        # print(outputs.logits)
        metric.add_batch(
            predictions=predictions,
            references=references,
        )
    # model.eval()
    # all_predictions = []
    # all_references = []

    # for step, batch in enumerate(tqdm(eval_dataloader)):
    #     batch = {k: v.to(device) for k, v in batch.items()}
    #     with torch.no_grad():
    #         outputs = model(**batch)

    #     # Extract predictions and references
    #     predictions = outputs.logits.argmax(dim=-1).cpu().numpy()  # Convert predictions to NumPy
    #     references = batch["labels"].cpu().numpy()  # Convert references to NumPy

    #     # Format predictions and references for multirc
    #     for idx, pred, ref in zip(batch["idx"], predictions, references):
    #         all_predictions.append({"idx": idx.item(), "prediction": int(pred)})
    #         all_references.append({"idx": idx.item(), "label": int(ref)})

    # # Add formatted predictions and references to the metric
    # metric.add_batch(
    #     predictions=all_predictions,
    #     references=all_references,
    # )


    eval_metric = metric.compute()
    if task == "stsb":
        acc_list.append(eval_metric['pearson'])
        log(f"epoch {epoch}:", eval_metric, ', current_best_pearson:',max(acc_list),'train_loss:',loss)
        print(f"epoch {epoch}:", eval_metric, '\033[32m, current_best_pearson:\033[0m',max(acc_list),'train_loss:',loss)
    elif task == 'cola':
        acc_list.append(eval_metric['matthews_correlation'])
        print(f"epoch {epoch}:", eval_metric, '\033[32m, current_best_corr:\033[0m',max(acc_list),'train_loss:',loss)
        log(f"epoch {epoch}:", eval_metric, ', current_best_corr:',max(acc_list),'train_loss:',loss)
    else:
        acc_list.append(eval_metric['accuracy'])
        print(f"epoch {epoch}:", eval_metric, '\033[32m, current_best_acc:\033[0m',max(acc_list),'train_loss:',loss)
        log(f"epoch {epoch}:", eval_metric, ', current_best_acc:',max(acc_list),'train_loss:',loss)
