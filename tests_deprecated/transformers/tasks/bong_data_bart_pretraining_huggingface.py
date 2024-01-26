import os
from itertools import chain

import math
import re
import nltk
import time
import torch
import numpy as np
import random
from tqdm.auto import tqdm
from typing import Dict, List, Optional

from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from datasets import Dataset as Dt
from datasets import DatasetDict, concatenate_datasets

from oslo.transformers.tasks.data_bart_pretraining import (
    ProcessorForBartPretraining,
    DataCollatorForBartPretraining,
)

from tests_deprecated.transformers.tasks.test_data_base import TestDataBinarization

from transformers import (
    BartTokenizerFast,
    BartConfig,
    BartModel,
    BartForConditionalGeneration,
    BartTokenizerFast,
    get_scheduler,
    PreTrainedTokenizerBase,
    BatchEncoding,
)
from transformers.models.bart.modeling_flax_bart import shift_tokens_right


try:
    from datasets import load_dataset
except ImportError:
    print("You have to install 'datasets' to test data_sequence_classification.py")


class FlaxDataCollatorForBartDenoisingLM:
    """
    Data collator used for BART denoising language modeling. The code is largely copied from
    `<https://github.com/morganmcg1/rotobart/blob/main/data_collator.py#L223>`__.
    For more information on how BART denoising language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.13461.pdf>`__
    or the `official code for preprocessing <https://github.com/facebookresearch/fairseq/blob/main/fairseq/data/denoising_dataset.py>`__ .
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data
        mask_ratio (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input
        poisson_lambda (:obj:`float`):
            Mean parameter of Poisson distribution used to generate span-lengths to be masked
        permute_sentence_ratio (:obj:`float`):
            Ratio of sentences to be permuted in each document
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        decoder_start_token_id: int,
        mask_ratio: float = 0.3,
        poisson_lambda: float = 3.0,
        permute_sentence_ratio: float = 1.0,
    ):

        self.tokenizer = tokenizer
        self.decoder_start_token_id = decoder_start_token_id
        self.mask_ratio = mask_ratio
        self.poisson_lambda = poisson_lambda
        self.permute_sentence_ratio = permute_sentence_ratio

    def __post_init__(self):
        if self.tokenizer.mask_token is None or self.tokenizer.eos_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token or eos token token which is necessary for denoising"
                " language modeling. "
            )

    def __call__(self, examples: List[Dict[str, List[int]]]) -> BatchEncoding:
        # convert list to dict and tensorize input
        batch = BatchEncoding(
            {
                k: np.array([examples[i][k] for i in range(len(examples))])
                for k, v in examples[0].items()
            }
        )
        batch["labels"] = batch["input_ids"].copy()
        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.tokenizer.pad_token_id, self.decoder_start_token_id
        )
        # permuting sentences
        do_permute = False
        if self.permute_sentence_ratio > 0.0:
            batch["input_ids"] = self.permute_sentences(batch["input_ids"])
            do_permute = True

        # masking span of tokens (text infilling in the paper)
        if self.mask_ratio:
            batch["input_ids"], batch["labels"] = self.span_mask_tokens(
                batch["input_ids"], batch["labels"], do_permute
            )

        # ignore pad tokens
        batch["attention_mask"] = (
            batch["input_ids"] != self.tokenizer.pad_token_id
        ).astype(int)
        batch["decoder_attention_mask"] = (
            batch["decoder_input_ids"] != self.tokenizer.pad_token_id
        ).astype(int)
        return batch

    def permute_sentences(self, input_ids):
        """
        Shuffle sentences in each document.
        """
        results = input_ids.copy()

        # find end locations of sentences
        # Example을 생성할때 문장이 들어오면 문장분리를 시행하는데 그때 문장사이에 PAD 토큰을 삽임함, 그토큰을 문장의 분리 기준으로 삼음.
        end_sentence_mask = input_ids == self.tokenizer.pad_token_id
        sentence_ends = np.argwhere(end_sentence_mask)
        sentence_ends[:, 1] += 1
        example_has_multiple_sentences, num_sentences = np.unique(
            sentence_ends[:, 0], return_counts=True
        )
        # permute_ratio를 곱하지 않은 각 batch index안에 SEP 분리된 문장의 수를 저장하고 있음 E.X. num_sentences_map[3] = 10 이면 batch_index 3에 10개의
        # 분리될 문장이 있다는 의미.
        num_sentences_map = {
            sent_idx: count
            for sent_idx, count in zip(example_has_multiple_sentences, num_sentences)
        }
        # num_to_permute 실제 분리될 문장이 10개라고 할때 전체중 permute_ratio만큼만 문장을 섞는다.
        num_to_permute = np.ceil(num_sentences * self.permute_sentence_ratio).astype(
            int
        )
        num_to_permute_map = {
            sent_idx: count
            for sent_idx, count in zip(example_has_multiple_sentences, num_to_permute)
        }
        # np.unique(sentence_ends[:, 0], return_index=True)[1][1:] < 각 batch가 시작되는 인덱스를 나타냄
        # [0, 19, 40] 이라면 0번째 인덱스에 example_has_multiple_sentences의 첫번째 값이 들위치하고, 19번째 인덱스에는 두번째 값이 위치하게된다
        # sentence_ends는 각 example들에서 분리될 문장의 인덱스를 담고 있다. E.X. sentence_ends[0] = array[20, 34, 86 ..] 20번째 34번째 86 번째에서 문장이 분리됨.
        sentence_ends = np.split(
            sentence_ends[:, 1],
            np.unique(sentence_ends[:, 0], return_index=True)[1][1:],
        )
        sentence_ends_map = {
            sent_idx: count
            for sent_idx, count in zip(example_has_multiple_sentences, sentence_ends)
        }

        for i in range(input_ids.shape[0]):
            if i not in example_has_multiple_sentences:
                continue
            # 총 문리될 문장을 랜덤으로 만들고(num_sentences_map), permute_ratio(num_to_permute_map)만큼 뽑아낸다
            # substitutions은 선택될 문장을 랜덤하게 고르는것이고, ordering[substitutions]은 선택된 문장을 랜덤하게 섞는 것이다.
            substitutions = np.random.permutation(num_sentences_map[i])[
                : num_to_permute_map[i]
            ]
            ordering = np.arange(0, num_sentences_map[i])
            # orderring에서 substitutions에 해당하는 인덱스를 num_to_permute_map을 랜덤으로 섞은 값으로 대체해서 문장을 섞어준다.
            ordering[substitutions] = substitutions[
                np.random.permutation(num_to_permute_map[i])
            ]

            # write shuffled sentences into results
            index = 0
            for j in ordering:
                # sentence_ends_map[i][j - 1] < i번째에는 문장들을 분리하기 위한 index가 저장 되어있고
                sentence = input_ids[
                    i,
                    (sentence_ends_map[i][j - 1] if j > 0 else 0) : sentence_ends_map[
                        i
                    ][j],
                ]
                results[i, index : index + sentence.shape[0]] = sentence
                index += sentence.shape[0]
        return results

    def span_mask_tokens(self, input_ids, labels, do_permute):
        """
        Sampling text spans with span lengths drawn from a Poisson distribution and masking them.
        """
        special_tokens_mask_labels = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        special_tokens_mask_inputs = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in input_ids.tolist()
        ]
        special_tokens_mask_labels = np.array(special_tokens_mask_labels, dtype=bool)
        special_tokens_mask_inputs = np.array(special_tokens_mask_inputs, dtype=bool)

        # determine how many tokens we need to mask in total
        is_token_mask = (
            ~(input_ids == self.tokenizer.pad_token_id) & ~special_tokens_mask_inputs
        )
        num_tokens_to_mask = int(
            math.ceil(is_token_mask.astype(float).sum() * self.mask_ratio)
        )
        if num_tokens_to_mask == 0:
            return input_ids, labels

        # generate a sufficient number of span lengths
        span_lengths = np.random.poisson(
            lam=self.poisson_lambda, size=(num_tokens_to_mask,)
        )
        while np.cumsum(span_lengths, 0)[-1] < num_tokens_to_mask:
            span_lengths = np.concatenate(
                [
                    span_lengths,
                    np.random.poisson(
                        lam=self.poisson_lambda, size=(num_tokens_to_mask,)
                    ),
                ]
            )

        # remove all spans of length 0
        # note that BART inserts additional mask tokens where length == 0,
        # which we do not implement for now as it adds additional complexity
        span_lengths = span_lengths[span_lengths > 0]

        # trim to about num_tokens_to_mask tokens
        cutoff_idx = (
            np.argmin(np.abs(np.cumsum(span_lengths, 0) - num_tokens_to_mask)) + 1
        )
        span_lengths = span_lengths[:cutoff_idx]

        # randomly choose starting positions for masking
        token_indices = np.argwhere(is_token_mask == 1)
        span_starts = np.random.permutation(token_indices.shape[0])[
            : span_lengths.shape[0]
        ]
        # prepare mask
        masked_indices = np.array(token_indices[span_starts])
        mask = np.full_like(input_ids, fill_value=False)

        # mask starting positions
        for mi in masked_indices:
            mask[tuple(mi)] = True
        span_lengths -= 1

        # fill up spans
        max_index = input_ids.shape[1] - 1
        remaining = (span_lengths > 0) & (masked_indices[:, 1] < max_index)
        while np.any(remaining):
            masked_indices[remaining, 1] += 1
            for mi in masked_indices:
                mask[tuple(mi)] = True
            span_lengths -= 1
            remaining = (span_lengths > 0) & (masked_indices[:, 1] < max_index)

        # place the mask tokens
        mask[np.where(special_tokens_mask_inputs)] = False
        input_ids[np.where(mask)] = self.tokenizer.mask_token_id
        if not do_permute:
            labels[np.where(mask == 0)] = -100
        else:
            labels[np.where(special_tokens_mask_labels)] = -100

        # remove mask tokens that are not starts of spans
        to_remove = (mask == 1) & np.roll((mask == 1), 1, 1)
        new_input_ids = np.full_like(input_ids, fill_value=self.tokenizer.pad_token_id)
        for i, example in enumerate(input_ids):
            new_example = example[~to_remove[i]]
            new_input_ids[i, : new_example.shape[0]] = new_example

        return new_input_ids, labels


def get_wiki_data():
    with open("wiki_output_file.txt", encoding="utf8") as f:
        wiki_texts = [line.replace("\n", "") for line in f.readlines()]
    dataset = DatasetDict({"train": Dt.from_dict({"text": wiki_texts})})
    return dataset


def get_cnn_data():
    dataset = load_dataset(
        # data_args.dataset_name, ############ !!!!!!!!!!!!!!!!!!
        "cnn_dailymail",
        "3.0.0",
        use_auth_token=None,
        # split = "train+validation+test"
        # use_auth_token=True if model_args.use_auth_token else None,
    )

    dataset = dataset.rename_column("article", "text")
    dataset = dataset.remove_columns(["highlights", "id"])
    # # reduce dataset for time
    dataset["train"] = Dt.from_dict(dataset["train"][:4523])
    # dataset["train"] = Dt.from_dict(dataset["train"][:])
    dataset["validation"] = Dt.from_dict(dataset["validation"][:555])
    dataset["test"] = Dt.from_dict(dataset["test"][:555])

    dataset = DatasetDict(
        {
            "train": concatenate_datasets(
                [dataset["train"], dataset["validation"], dataset["test"]]
            )
        }
    )

    return dataset


if "__main__" == __name__:

    # GPU settings
    # world_size = int(os.environ["WORLD_SIZE"])
    # rank = int(os.environ["LOCAL_RANK"])
    # print(f'rank = {rank} world_size = {world_size} ')
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.cuda.current_device()

    # dist.init_process_group(bacskend="nccl",world_size=world_size, rank=rank)
    # torch.cuda.set_device(rank)
    # device = torch.cuda.current_device()
    print(f"rank = {rank} world_size = {world_size} device = {device}")
    dataset_name = "wiki"  # wiki or cnn
    dataset = get_cnn_data() if dataset_name == "cnn" else get_wiki_data()

    model_nm = "facebook/bart-base"
    max_seq_length = 512

    # Model Config Setting
    config = BartConfig(
        vocab_size=50265,
        max_position_embeddings=1024,
        d_model=1024,
        decoder_attention_heads=16,
        decoder_ffn_dim=4096,
        decoder_layers=12,
        decoder_start_token_id=2,
        dropout=0.1,
        encoder_attention_heads=16,
        encoder_ffn_dim=4096,
        encoder_layers=12,
        num_hidden_layers=6,
    )
    # 실험의 재현(reproducible)을 위한 seed 셋팅
    seed_num = 1993
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)  # if use multi-GPU
    tokenizer = BartTokenizerFast.from_pretrained(model_nm)

    ### Data Processor ###
    tokenized_datasets = dataset.load_from_disk(
        f"/home/bsj/.cache/huggingface/datasets/{dataset_name}_processed_huggingface"
    )

    # preprocessing_num_workers = 50
    # nltk.download("punkt")
    # sentence_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    # def sentence_split_function(example):
    #     sents = sentence_tokenizer.tokenize(example["text"])
    #     # use pad token as end of sentence indicator
    #     new_text = tokenizer.bos_token + f"{tokenizer.pad_token}".join(sents) + tokenizer.eos_token
    #     return {"text": new_text}
    # # cache_nm = {'train':'train_test.arrow', 'test':'test_test.arrow', 'validation':'validation_test.arrow'}
    # split_datasets = dataset.map(
    #     sentence_split_function,
    #     batched=False,
    #     num_proc= preprocessing_num_workers,
    #     remove_columns=None,
    #     load_from_cache_file=not None,
    # )

    # # Tokenize every text, then concatenate them together before splitting them in smaller parts.
    # # Since we make sure that all sequences are of the same length, no attention_mask is needed.
    # def tokenize_function(examples):
    #     return tokenizer(examples['text'], add_special_tokens=False, return_attention_mask=False)

    # tokenized_datasets = split_datasets.map(
    #     tokenize_function,
    #     batched=True,
    #     num_proc= preprocessing_num_workers,
    #     remove_columns='text',
    #     load_from_cache_file=not None,

    # )
    # # Main data processing function that will concatenate all texts from our dataset and generate chunks of
    # # max_seq_length.
    # def group_texts(examples):
    #     # Concatenate all texts.
    #     concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    #     total_length = len(concatenated_examples[list(examples.keys())[0]])
    #     # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    #     # customize this part to your needs.
    #     if total_length >= max_seq_length:
    #         total_length = (total_length // max_seq_length) * max_seq_length
    #     # Split by chunks of max_len.
    #     result = {
    #         k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
    #         for k, t in concatenated_examples.items()
    #     }
    #     return result

    # tokenized_datasets = tokenized_datasets.map(
    #     group_texts,
    #     batched=True,
    #     num_proc= preprocessing_num_workers,
    #     load_from_cache_file=not None,
    # )
    ### Data Processor end ###

    # tokenized_datasets.save_to_disk(f'/home/bsj/.cache/huggingface/datasets/{dataset_name}_processed_huggingface')
    processed_dataset = tokenized_datasets["train"]

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = FlaxDataCollatorForBartDenoisingLM(
        tokenizer=tokenizer,
        decoder_start_token_id=tokenizer.eos_token_id,
        mask_ratio=0.3,
        poisson_lambda=3.0,
        permute_sentence_ratio=1.0,
    )

    if data_collator.tokenizer.pad_token is None:
        data_collator.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        tokenizer = data_collator.tokenizer
        print("pad_token is set.")

    # for Distributed Data Parallel
    per_device_train_batch_size = 36
    strategy = "ddp"  # ddp or None
    dataloader_worker = 3 * 4
    train_sampler = (
        DistributedSampler(
            processed_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        if strategy == "ddp"
        else None
    )
    train_dataloader = DataLoader(
        dataset=processed_dataset,
        batch_size=per_device_train_batch_size,
        shuffle=False,
        collate_fn=data_collator,
        sampler=train_sampler,
        num_workers=dataloader_worker,
        pin_memory=True,
    )

    # np.random.seed(seed_num), #random.seed(seed_num), #torch.cuda.manual_seed(random_seed), #torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    model = BartForConditionalGeneration(config).cuda()
    model = DistributedDataParallel(model, device_ids=[device], output_device=device)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    weight_decay = 0.02
    learning_rate = 5e-5
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            # Parameters that are not bias and LayerNorm.weight will receive a penalty
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=learning_rate)

    check_val_every_n_epoch = 1
    max_epochs = 5
    warmup_ratio = 0.3  # for scheduler
    # num_train_steps_per_epoch = int(len(train_dataloader) / (per_device_train_batch_size )) + int(len(val_dataloader) / (per_device_val_batch_size )) * check_val_every_n_epoch
    num_train_steps_per_epoch = int(
        len(processed_dataset) / (per_device_train_batch_size)
    )
    num_train_steps_per_epoch = (
        num_train_steps_per_epoch // world_size
    )  # 멀티 프로세싱에서 GPU가 여러대 사용될 경우 데이터를 나눠서 학습하기 때문에 이와 같이 진행함.
    total_num_train_steps = num_train_steps_per_epoch * max_epochs

    num_warmup_steps = int(total_num_train_steps * warmup_ratio)
    gradient_accumulation_steps = 1

    # linear ->  get_linear_schedule_with_warmup,
    # cosine ->  get_cosine_schedule_with_warmup
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_training_steps=total_num_train_steps,
        num_warmup_steps=num_warmup_steps * gradient_accumulation_steps,
    )

    import wandb

    # wandb
    wandb.init(
        project="bart_pretraining-project_huggingface",
        config={
            "dataset": f"{dataset_name}",
            "epochs": max_epochs,
            "learning-rate": learning_rate,
        },
    )
    ### Training Start ###
    train_dataloader_len = len(train_dataloader)
    global_count = 0
    model.train()

    from torch.cuda.amp import GradScaler

    scaler = GradScaler()

    for epoch in range(max_epochs):

        epoch_tqdm_dataloader = tqdm(
            train_dataloader,
            f"Training( {epoch} / {max_epochs} ) ",
            dynamic_ncols=True,
        )
        for i, batch in enumerate(epoch_tqdm_dataloader):
            global_count += 1
            optimizer.zero_grad()
            batch = {k: torch.tensor(v, device=device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()

            if i % 100 == 0:
                epoch_tqdm_dataloader.set_postfix(
                    {"loss": loss, "global_count": global_count}
                )
                wandb.log({"loss": loss})
            # model save
            if global_count % 10000 == 0:
                if rank == 0:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": loss,
                        },
                        f"outputs/huggingface_kobart_pre_epoch{epoch}_global_{global_count}_loss={loss:.5f}.pt",
                    )

    wandb.finish()

# model load end generate
