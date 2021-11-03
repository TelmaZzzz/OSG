import torch
from config import *
from utils import utils
import logging
# from torch._C import dtype
# from transformers.utils.dummy_pt_objects import BartForCausalLM, BartModel
from transformers import AutoTokenizer, BartForConditionalGeneration, BertTokenizer, T5Tokenizer, T5ForConditionalGeneration
from modeling_cpt import CPTForConditionalGeneration
from train import *
import json
import metrics
import eval
import random
import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
logging.getLogger().setLevel(logging.INFO)


class Example(object):
    def __init__(self, story, outline, title=""):
        self.story = story
        self.outline = outline
        self.un_cat_story = self.story
        if len(self.story) >= 505:
            self.story = self.story[:500]
        # self.title = title


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, Examples, tokenizer):
        super(BaseDataset, self).__init__()
        self.input_ids = []
        self.input_mask = []
        self.output_ids = []
        self.output_mask = []
        self.build(Examples, tokenizer)
 
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "input_mask": self.input_mask[idx],
            "output_ids": self.output_ids[idx],
            "output_mask": self.output_mask[idx]
        }
    
    def __len__(self):
        return len(self.input_ids)
    
    def build(self, Examples, tokenizer):
        for item in Examples:
            input = "[word]"
            # input = ""
            for word in item.outline:
                input += "<w>"
                input += word
            input += "[SEP]"
            output = "[CLS]" + item.story + "[SEP]"
            # output = item.story + "[SEP]"
            input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input))
            input_mask = [1] * len(input_ids)
            output_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(output))
            output_mask = [1] * len(input_ids)
            self.input_ids.append(input_ids)
            self.input_mask.append(input_mask)
            self.output_ids.append(output_ids)
            self.output_mask.append(output_mask)


class Collection(object):
    def __init__(self, args):
        self.config = {}
        self.config["BUCKET"] = True
        self.config["FIX_LENGTH"] = args.fix_length
        self.config["PAD_ID"] = args.pad_id

    def __call__(self, batch):
        out = {
            "input_ids": [],
            "input_mask": [],
            "output_ids": [],
            "output_mask": [],
        }
        for mini_batch in batch:
            for k, v in mini_batch.items():
                out[k].append(v)
        input_max_pad = 0
        output_max_pad = 0
        if self.config["BUCKET"]:
            for p in out["input_ids"]:
                input_max_pad = max(input_max_pad, len(p))
            for p in out["output_ids"]:
                output_max_pad = max(output_max_pad, len(p))
        else:
            input_max_pad = self.config["FIX_LENGTH"]
            output_max_pad = self.config["FIX_LENGTH"]
        for i in range(len(batch)):
            out["input_ids"][i] = out["input_ids"][i] + [self.config["PAD_ID"]] * (input_max_pad - len(out["input_ids"][i]))
            out["input_mask"][i] = out["input_mask"][i] + [0] * (input_max_pad - len(out["input_mask"][i]))
            out["output_ids"][i] = out["output_ids"][i] + [-100] * (output_max_pad - len(out["output_ids"][i]))
            out["output_mask"][i] = out["output_mask"][i] + [0] * (output_max_pad - len(out["output_mask"][i]))
        out["input_ids"] = torch.tensor(out["input_ids"], dtype=torch.long)
        out["input_mask"] = torch.tensor(out["input_mask"], dtype=torch.long)
        out["output_ids"] = torch.tensor(out["output_ids"], dtype=torch.long)
        out["output_mask"] = torch.tensor(out["output_mask"], dtype=torch.long)
        return out 


def read_json(path):
    data = utils.read_data(path)
    res = []
    for item in data:
        # utils.debug("item", item)
        res.append(json.loads(item))
    return res


def prepare_examples(path):
    data = read_json(path)
    Examples = []
    for item in data:
        Examples.append(Example(story=item["story"], outline=item["outline"]))
    return Examples


def main(args):
    logging.getLogger().setLevel(logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logging.info("Config Init")
    torch.cuda.set_device(args.local_rank)
    if args.local_rank != -1:
        dist.init_process_group(backend='nccl')
    args.device = torch.device("cuda", args.local_rank if args.local_rank != -1 else 0)
    logging.info("Load Data")
    train_data = prepare_examples(args.train_path)
    valid_data = prepare_examples(args.valid_path)
    args.gold = []
    args.outline = []
    for item in valid_data:
        args.gold.append(item.un_cat_story)
        args.outline.append(item.outline)
    # test_data = prepare_examples(args.test_path)
    logging.info("Init Model and Tokenizer")
    args.n_gpu = torch.cuda.device_count()
    utils.debug("tokenizer", args.tokenizer_path)
    utils.debug("pretrain", args.pretrain_path)
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)
    # tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_path)
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    # tokenizer = BartTokenizer.from_file(args.tokenizre_path)
    if args.model_load:
        model = torch.load(args.model_load)
    else:
        model = BartForConditionalGeneration.from_pretrained(args.pretrain_path)
        # model = T5ForConditionalGeneration.from_pretrained(args.pretrain_path)
    # utils.debug("model", model)
    # model = CPTForConditionalGeneration.from_pretrained(args.pretrain_path)
    # special_token = {"additional_special_tokens": ["[titile]"] + ["EOS"] + ["BOS"] + [f"<w{i}>" for i in range(8)]}
    special_token = {"additional_special_tokens": ["[titile]"] + ["[SEP]"] + ["[CLS]"] + ["[word]"] + ["<w>"]}
    tokenizer.add_special_tokens(special_token)
    word_token = ["“", "”"]
    tokenizer.add_tokens(word_token)
    tokenizer.pad_token = "[PAD]"
    tokenizer.eos_token = "[SEP]"
    tokenizer.bos_token = "[CLS]"
    # tokenizer.eos_token = "[eos]"
    # tokenizer.bos_token = "[bos]"
    model.config.decoder_start_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.forced_eos_token_id = tokenizer.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    model.config.device = args.device
    logging.info(f"eos_token_id:{model.config.eos_token_id}")
    logging.info(f"bos_token_id:{model.config.bos_token_id}")
    logging.info(f"gpu num:{args.n_gpu}")
    # DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    logging.info(f"local rank:{args.local_rank}")
    model = model.to(args.device)
    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    # if args.n_gpu > 1:
    #     model = DDP(model).
    # else:
    #     model = model.cuda()
    args.pad_id = tokenizer.pad_token_id
    logging.info("Prepare Dataset")
    train_dataset = BaseDataset(train_data, tokenizer)
    valid_dataset = BaseDataset(valid_data, tokenizer)
    if args.local_rank == -1:
        train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=Collection(args))
        valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=Collection(args))
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        valid_sampler = utils.SequentialDistributedSampler(valid_dataset, args.batch_size)
        args.valid_len = len(valid_dataset)
        # test_dataset = BaseDataset(test_data, tokenizer)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=Collection(args))
        valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, sampler=valid_sampler, collate_fn=Collection(args))
    # test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=Collection(args))
    args.parameter = utils.get_train_parameter()
    logging.info("Start Training")
    Base_train(train_iter, valid_iter, model, tokenizer, args)


def predict(args):
    logging.info("Config Init")
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl')
    args.device = torch.device("cuda", args.local_rank)
    logging.info("Load Data")
    test_data = prepare_examples(args.test_path, is_train=False)
    args.gold = []
    args.outline = []
    for item in test_data:
        args.gold.append(item.un_cat_story)
        args.outline.append(item.outline)
    # test_data = prepare_examples(args.test_path)
    logging.info("Init Model and Tokenizer")
    args.n_gpu = torch.cuda.device_count()
    utils.debug("tokenizer", args.tokenizer_path)
    utils.debug("pretrain", args.pretrain_path)
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    # tokenizer = BartTokenizer.from_file(args.tokenizre_path)
    model = torch.load(args.model_load).to(args.device)
    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    # utils.debug("model", model)
    # model = CPTForConditionalGeneration.from_pretrained(args.pretrain_path)
    # special_token = {"additional_special_tokens": ["[titile]"] + ["EOS"] + ["BOS"] + [f"<w{i}>" for i in range(8)]}
    special_token = {"additional_special_tokens": ["[titile]"] + ["[SEP]"] + ["[CLS]"] + ["[word]"] + ["<w>"]}
    tokenizer.add_special_tokens(special_token)
    word_token = ["“", "”"]
    tokenizer.add_tokens(word_token)
    # tokenizer.pad_token = "[PAD]"
    tokenizer.eos_token = "[SEP]"
    tokenizer.bos_token = "[CLS]"
    args.pad_id = tokenizer.pad_token_id
    test_dataset = BaseDataset(test_data, tokenizer)
    args.test_len = len(test_dataset)
    test_sampler = utils.SequentialDistributedSampler(test_dataset, args.batch_size)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, collate_fn=Collection(args))
    logging.info("Start predict")
    parameter_list = utils.get_parameter()
    continue_list = []
    args.output += f"_batch{args.batch_size}"
    with torch.no_grad():
        for idx, parameter in enumerate(parameter_list):
            if idx in continue_list:
                continue
            args.parameter = parameter
            args.step = idx
            Base_predict(test_iter, model, tokenizer, args)
    logging.info("END")


if __name__ == "__main__":
    args = Base_config()
    utils.set_seed(959794+args.local_rank)
    if args.train:
        args.model_save = '/'.join([args.model_save, utils.d2s(datetime.datetime.now(), time=True)])
        main(args)
    if args.predict:
        args.output = '/'.join([args.output, utils.d2s(datetime.datetime.now(), time=True)])
        predict(args)