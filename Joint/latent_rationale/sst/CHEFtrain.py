import os
import re
import time, datetime
import json
from collections import OrderedDict, Counter
from tqdm import tqdm

import torch
import torch.optim
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import (
    TensorDataset,
    Dataset,
    random_split, 
    Subset,
    DataLoader,
    RandomSampler,
    SequentialSampler
)
from torch import nn
from transformers import (
    AutoModel,
    AutoTokenizer
)
from sklearn.metrics import (
    f1_score, 
    precision_recall_fscore_support,
    confusion_matrix
)

from latent_rationale.common.util import make_kv_string
from latent_rationale.sst.vocabulary import Vocabulary
from latent_rationale.sst.models.model_helpers import build_model, build_CHEFmodel
from latent_rationale.sst.util import get_args, sst_reader, \
    prepare_minibatch, get_minibatch, load_glove, print_parameters, \
    initialize_model_, get_device
from latent_rationale.sst.evaluate import evaluate

import datetime
import os
import json
class Logger:
    def __init__(self, log_path=None, config=None, reopen_to_flush=False):
        self.log_file = None
        self.reopen_to_flush = reopen_to_flush
        if log_path is not None:
            os.makedirs(os.path.dirname(os.path.join(log_path, 'log.txt')), exist_ok=True)
            self.log_file = open(os.path.join(log_path, 'log.txt'), 'a+')
        if config is not None:
            with open(os.path.join(log_path, 'log_config.json'), 'w') as f:
                json.dump(config, f, indent=4)

    def log(self, msg):
        formatted = f'[{datetime.datetime.now().replace(microsecond=0).isoformat()}] {msg}'
        print(formatted)
        if self.log_file:
            self.log_file.write(formatted + '\n')
            if self.reopen_to_flush:
                log_path = self.log_file.name
                self.log_file.close()
                self.log_file = open(log_path, 'a+')
            else:
                self.log_file.flush()


# python -m debugpy --listen 5556 --wait-for-client -m latent_rationale.sst.CHEFtrain --model latent --selection 0.3 --save_path results/chef/


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


cfg = get_args()
cfg = vars(cfg)

logger = Logger(os.path.join("logdir", cfg['logdir']), cfg)
# import ipdb; ipdb.set_trace()

device = get_device()
logger.log("device:{}".format(device))


def train():
    """
    Main training loop.
    """

    for k, v in cfg.items():
        logger.log("{:20} : {:10}".format(k, v))

    num_iterations = cfg["num_iterations"]
    print_every = cfg["print_every"]
    eval_every = cfg["eval_every"]
    batch_size = cfg["batch_size"]
    eval_batch_size = cfg.get("eval_batch_size", batch_size)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    logger.log("Loading data")
    train_dataset, test_dataset = loadData('data', tokenizer)

    logger.log("train {}".format(len(train_dataset)))
    logger.log("test {}".format(len(test_dataset)))
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=1
    )
    val_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=1
    )
    
    # iters_per_epoch = len(train_data) // cfg["batch_size"]

    # if cfg["eval_every"] == -1:
    #     eval_every = iters_per_epoch
    #     print("Set eval_every to {}".format(iters_per_epoch))

    # if cfg["num_iterations"] < 0:
    #     num_iterations = iters_per_epoch * -1 * cfg["num_iterations"]
    #     print("Set num_iterations to {}".format(num_iterations))

    writer = SummaryWriter(log_dir=cfg["save_path"])  # TensorBoard

    # Build model
    model = build_CHEFmodel('bert-base-chinese', cfg)
    # initialize_model_(model)

    # with torch.no_grad():
    #     model.embed.weight.data.copy_(torch.from_numpy(vectors))
    #     if cfg["fix_emb"]:
    #         print("fixed word embeddings")
    #         model.embed.weight.requires_grad = False
    #     model.embed.weight[1] = 0.  # padding zero

    optimizer = Adam(model.parameters(), lr=cfg["lr"],
                     weight_decay=cfg["weight_decay"])

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg["lr_decay"], patience=cfg["patience"],
        verbose=True, cooldown=cfg["cooldown"], threshold=cfg["threshold"],
        min_lr=cfg["min_lr"])
    criterion = nn.CrossEntropyLoss()

    iter_i = 0
    train_loss = 0.
    print_num = 0
    start = time.time()
    losses = []
    accuracies = []
    best_eval = 1.0e9
    best_iter = 0

    model = model.to(device)

    # print model
    # print(model)
    # print_parameters(model)
    epochs = 30
    for epoch_i in range(0, epochs):
        logger.log('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        logger.log('Training...')
        # Measure how long the training epoch takes.
        t0 = time.time()
        train_loss = 0

        model.train()
        # ========================================
        #               Training
        # ========================================
        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                # Report progress.
                logger.log('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}. train loss{}'.format(step, len(train_dataloader), elapsed, train_loss/step))

            logits = model(batch)
            loss, loss_optional = model.get_loss(logits, batch[2])

            model.zero_grad()
            train_loss += loss.item()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(
            #     model.parameters(),
            #     max_norm=cfg["max_grad_norm"]
            # )
            optimizer.step()
        logger.log(f'Train_loss is {train_loss}')
        # ========================================
        #               Validation
        # ========================================
        t0 = time.time()
        # Put the model in evaluation mode
        model.eval()
        all_predictions = torch.tensor([])
        all_labels = torch.tensor([])
        validate_loss = 0.0
        for batch in val_dataloader:
            with torch.no_grad():
                logits = model(batch)
            # print(logits)
            predictions = model.predict(logits)

            loss, loss_optional = model.get_loss(logits, batch[2])
            validate_loss += loss.item()
            all_predictions = torch.cat([all_predictions, predictions.detach().cpu()])
            all_labels = torch.cat([all_labels, batch[2].detach().cpu()])

        # logger.log(all_predictions)
        # logger.log(all_labels)
        # scheduler.step(validate_loss)
        c = Counter()
        for pred in all_predictions:
            c[int(pred.item())] += 1
        logger.log(c)
        c = Counter()
        for label in all_labels:
            c[int(label.item())] += 1
        logger.log(c)
        pre, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='micro')
        logger.log("       F1 (micro): {:.2%}".format(f1))
        pre, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro')
        logger.log("Precision (macro): {:.2%}".format(pre))
        logger.log("   Recall (macro): {:.2%}".format(recall))
        logger.log("       F1 (macro): {:.2%}".format(f1))


def loadData(filepath, tokenizer):
    """
    :param filepath: data file path
    :param tokenizer
    :return: train test set
    """
    data_list = json.load(open(f'{filepath}/test.json', 'r', encoding='utf-8')) \
        + json.load(open(f'{filepath}/train.json', 'r', encoding='utf-8')) 
    # for debug
    # indices = json.load(open('data/label_indices.json', 'r', encoding='utf-8'))
    # indices = [ind[:10] for ind in indices]
    # indices = indices[0] + indices[1] + indices[2]
    # data_list = [data_list[i] for i in indices]
    labels = []
    claims_ids = []
    evidences_ids_list = []
    for row in tqdm(data_list):
        labels.append(row['label'])
        encoded_dict = tokenizer.encode_plus(
            row.get('claim', row.get('title', '')),  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=128,  # Pad & truncate all sentences.
            padding='max_length',
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
            truncation=True
        )
        claims_ids.append(encoded_dict['input_ids'])
        # split the origin evidence text to sentences
        evidences = []
        # evidences += re.split(r'[？：。！（）.“”…\t\n]', row['content'])
        for ev in row.get('evidence', {}).values():
            evidences += re.split(r'[？：。！（）.“”…\t\n]', ev['text'])
        evidences_ids = []
        for evidence in evidences:
            encoded_dict = tokenizer.encode_plus(
                evidence,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=128,  # Pad & truncate all sentences.
                padding='max_length',
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
                truncation=True
            )
            evidences_ids.append(encoded_dict['input_ids'])
        evidences_ids = torch.cat(evidences_ids, dim=0).to(device)
        evidences_ids_list.append(evidences_ids)
    labels = torch.tensor(labels).to(device)
    # import ipdb;
    # ipdb.set_trace()
    claims_ids = torch.cat(claims_ids, dim=0).to(device)

    dataset = CHEFDataset(claims_ids, evidences_ids_list, labels)
    test_dataset, train_dataset = Subset(dataset, [i for i in range(999)]),\
        Subset(dataset, [i for i in range(999, len(dataset))])
    # train_len = int(len(dataset) * 0.8)
    # train_dataset, test_dataset = random_split(dataset, [train_len, len(dataset)-train_len])
    return train_dataset, test_dataset


class CHEFDataset(Dataset):
    def __init__(self, claims_ids, evidences_ids_list, labels):
        self.claims_ids = claims_ids    # tensor
        self.evidences_ids_list = evidences_ids_list # list[tensor...]
        self.labels = labels            # tensor

    def __getitem__(self, i):
        return (
            self.claims_ids[i], 
            self.evidences_ids_list[i],
            self.labels[i]
        )
    
    def __len__(self):
        return len(self.claims_ids)


if __name__ == "__main__":
    train()
