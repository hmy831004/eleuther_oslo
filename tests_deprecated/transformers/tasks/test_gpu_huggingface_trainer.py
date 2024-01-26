from datasets import load_dataset

nsmc = load_dataset('nsmc')

# train_data = nsmc['train'].shuffle(seed=42).select(range(12000))
# test_data = nsmc['test'].shuffle(seed=42).select(range(12000))
train_data = nsmc['train'].shuffle(seed=42)
test_data = nsmc['test'].shuffle(seed=42)

MODEL_NAME = 'bert-base-multilingual-cased'
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_encoding = tokenizer(
    train_data['document'],
    return_tensors='pt',
    padding=True,
    truncation=True
)

test_encoding = tokenizer(
    test_data['document'],
    return_tensors='pt',
    padding=True,
    truncation=True
)

import torch
from torch.utils.data import Dataset


class NSMCDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encoding = encodings
        self.labels = labels

    def __getitem__(self, idx):
        data = {key: val[idx] for key, val in self.encoding.items()}
        data['labels'] = torch.tensor(self.labels[idx]).long()

        return data

    def __len__(self):
        return len(self.labels)
train_set = NSMCDataset(train_encoding, train_data['label'])
test_set = NSMCDataset(test_encoding, test_data['label'])

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir = './outputs', # model이 저장되는 directory
    logging_dir = './logs', # log가 저장되는 directory
    logging_strategy ='no',
    num_train_epochs = 100, # training epoch 수
    per_device_train_batch_size=256,  # train batch size
    per_device_eval_batch_size=256,   # eval batch size
    logging_steps = 1000, # logging step, batch단위로 학습하기 때문에 epoch수를 곱한 전체 데이터 크기를 batch크기로 나누면 총 step 갯수를 알 수 있다.
    save_steps= 1000, # 50 step마다 모델을 저장한다.
    save_total_limit=2, # 2개 모델만 저장한다.3
    report_to="none"

)

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

from datasets import load_metric

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    m1 = load_metric('accuracy')
    m2 = load_metric('f1')

    acc = m1.compute(predictions=preds, references=labels)['accuracy']
    f1 = m2.compute(predictions=preds, references=labels)['f1']

    return {'accuracy':acc, 'f1':f1}

model.to(device)

# Trainer 에서 _setup_devices GPU init_process진행
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set, # 학습 세트
    eval_dataset=test_set, # 테스트 세트
    compute_metrics=compute_metrics, # metric 계산 함수,
)
# DistributedModelParallel , DistributedSampler선언 여기서함.
trainer.train()