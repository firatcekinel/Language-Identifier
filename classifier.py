from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset, DatasetDict, load_dataset, Features, Value
import evaluate
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from huggingface_hub import login
import torch
import wandb
from tqdm import tqdm
import argparse
import time
from dotenv import load_dotenv
import os


def login_credentials():
    # Get credentials
    hf_token = os.getenv("HF_TOKEN")
    wandb_api_key = os.getenv("WANDB_API_KEY")

    # Login to Hugging Face Hub
    if hf_token:
        login(token=hf_token)
    else:
        print("Hugging Face token not found.")

    # Login to Weights & Biases
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    else:
        print("W&B API key not found.")


# Load environment variables from the .env file
load_dotenv()
login_credentials()

# answerdotai/ModernBERT-large
parser = argparse.ArgumentParser()
parser.add_argument(
    '--model_id', default='FacebookAI/xlm-roberta-large', type=str, help='model id')
parser.add_argument('--cache_dir', default='/dl_scratch1/frtcek/.cache',
                    type=str, help='model cache dir')
parser.add_argument('--skip_train', default=False,
                    type=bool, help='skip to inference')
parser.add_argument('--skip_loading', default=True,
                    type=bool, help='skip loading dataset')
parser.add_argument('--n_epoch', default=2,
                    type=int, help='number of epochs')
parser.add_argument('--wandb_proj_name', default="lid",
                    type=str, help='wandb project name')
args = parser.parse_args()

model_id = args.model_id
model_name = model_id.split("/")[-1]
cache_dir = args.cache_dir
skip_train = args.skip_train
output_dir = "outputs/"
wandb_proj_name = args.wandb_proj_name
file_name = f"{model_name}_OpenLID-v2"

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

tokenizer = AutoTokenizer.from_pretrained(
    model_id, cache_dir=cache_dir, add_prefix_space=True)
tokenizer.model_max_length = 512
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    acc = accuracy.compute(predictions=predictions, references=labels)
    f1_macro = f1.compute(predictions=predictions,
                          references=labels, average="macro")
    return {"accuracy": acc, "f1-macro": f1_macro}


if args.skip_loading:
    train_df = pd.read_csv("data/train.csv", encoding="utf-8", sep="\t")
    train_df = train_df[["text", "label"]]
    val_df = pd.read_csv("data/val.csv", encoding="utf-8", sep="\t")
    val_df = val_df[["text", "label"]]
    test_df = pd.read_csv("data/test.csv", encoding="utf-8", sep="\t")
    test_df = test_df[["text", "label"]]

    val_df = val_df.groupby('label', group_keys=False).apply(lambda x: x.sample(min(len(x), 50), random_state=42))
    test_df = test_df.groupby('label', group_keys=False).apply(lambda x: x.sample(min(len(x), 100), random_state=42))

else:
    features = Features({
        "text": Value("string"),
        "language": Value("string"),
        "source": Value("string"),
        "__index_level_0__": Value("int64")
    })
    dataset = load_dataset("laurievb/OpenLID-v2",
                           cache_dir=args.cache_dir, trust_remote_code=True, features=features)

    dataset = dataset.remove_columns(["__index_level_0__", "source"])
    dataset = dataset.rename_column("language", "label")

    train_df = dataset["train"].to_pandas()

    train_df, val_df = train_test_split(
        train_df, test_size=0.002, random_state=42, stratify=train_df["label"])
    val_df, test_df = train_test_split(
        val_df, test_size=0.5, random_state=42, stratify=val_df["label"])

    train_df = train_df.groupby('label', group_keys=False).apply(
        lambda x: x.sample(min(len(x), 200), random_state=42))

    train_df.to_csv("data/train.csv", encoding="utf-8", sep="\t")
    val_df.to_csv("data/val.csv", encoding="utf-8", sep="\t")
    test_df.to_csv("data/test.csv", encoding="utf-8", sep="\t")

lid_labels = set(train_df["label"])
id2label = {i: label for i, label in enumerate(lid_labels)}
label2id = {v: k for k, v in id2label.items()}
all_labels = list(id2label.keys())

print(f"""train size: {train_df.shape[0]}, 
val size: {val_df.shape[0]},
test size: {test_df.shape[0]}""")

train_df["label"] = train_df["label"].map(label2id)
val_df["label"] = val_df["label"].map(label2id)
test_df["label"] = test_df["label"].map(label2id)


dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "validation": Dataset.from_pandas(val_df),
    "test": Dataset.from_pandas(test_df),
})

tokenized_dataset = dataset.map(
    preprocess_function, batched=True).remove_columns(["text"])


def init_model(config=None):

    # initialize a new wandb run
    with wandb.init(config=config, project=wandb_proj_name) as run:

        # if called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        run_name = f"task:lid|model_id:{model_name}|batch:{config.batch_size}|lr:{config.learning_rate}"
        params = {
            "batch_size": config.batch_size,
            "lr": config.learning_rate,
            "run_name": run_name
        }

        run.name = run_name
        train(params)

        torch.cuda.empty_cache()
        time.sleep(2)


def train(params):
    global sweep_no
    sweep_no += 1

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=len(all_labels), id2label=id2label, label2id=label2id, cache_dir=cache_dir
    ).to("cuda")

    training_args = TrainingArguments(
        output_dir=output_dir + model_name + "_" + str(sweep_no),
        optim="adamw_torch_fused",
        logging_steps=1,
        learning_rate=params["lr"],
        per_device_train_batch_size=params["batch_size"],
        per_device_eval_batch_size=params["batch_size"],
        num_train_epochs=args.n_epoch,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=0.2,
        eval_steps=0.1,
        report_to="wandb",
        run_name=params["run_name"],
        load_best_model_at_end=True,
        seed=42,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=False,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model()

    evaluate(trainer.model, params["run_name"])


def evaluate(model, run_name):
    global result_dict
    global test_df

    references = test_df["label"].to_list()
    predictions = []

    for idx, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
        inputs = tokenizer(row["text"], return_tensors="pt",
                           truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits
        y_pred = logits.argmax().item()
        predictions.append(y_pred)

    print(classification_report(references, predictions, labels=all_labels))

    conf_matrix = str(confusion_matrix(references, predictions).tolist())
    metrics = {
        "f1-macro": f1_score(references, predictions, average='macro'),
        "f1-micro": f1_score(references, predictions, average='micro'),
        "f1-weighted": f1_score(references, predictions, average='weighted'),
        "precision": precision_score(references, predictions, average='weighted'),
        "recall": recall_score(references, predictions, average='weighted'),
        "accuracy": accuracy_score(references, predictions)
    }

    #result_dict["confusion matrix"].append(conf_matrix)
    result_dict["metrics"].append(metrics)
    result_dict["run name"].append(run_name)
    result_dict["f1-macro"].append(metrics["f1-macro"])


result_dict = {"run name": [], "metrics": [],
                "f1-macro": []}
sweep_no = 0

if skip_train:
    for i in range(1, 9):
        model = AutoModelForSequenceClassification.from_pretrained(output_dir + model_name + "_" + str(
            i), num_labels=len(all_labels), id2label=id2label, label2id=label2id, cache_dir=cache_dir)
        evaluate(model, run_name=None)

    result_df = pd.DataFrame.from_dict(result_dict)
    print(result_df)

    result_df.to_csv(file_name + ".csv", sep="\t",
                     encoding="utf-8",  mode='a', header=False)

else:
    sweep_config = {
        'method': 'grid',
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'learning_rate': {
                'values': [1e-5, 2e-5, 5e-5]
            },
            'batch_size': {
                'values': [8, 16]
            },
        }
    }

    sweep_id = wandb.sweep(sweep_config, project=wandb_proj_name)

    wandb.agent(sweep_id, function=init_model, count=6)

    result_df = pd.DataFrame.from_dict(result_dict)
    print(result_df)

    result_df.to_csv(file_name + ".csv", sep="\t",
                     encoding="utf-8",  mode='a', header=False)
