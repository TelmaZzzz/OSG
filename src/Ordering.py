from os import WSTOPSIG
import torch
from torch._C import dtype
from transformers.utils.dummy_pt_objects import BartForCausalLM, BartModel
from config import *
from utils import utils
import logging
import json
from transformers import AutoTokenizer, BartForConditionalGeneration
from train import *
import metrics
import random
import datetime
logging.getLogger().setLevel(logging.INFO)


class Raw_Example(object):
    def __init__(self, story, outline, title=""):
        self.story = story
        self.outline = outline
        self.title = title
    
    def build_order(self):
        self.order = [0] * len(self.outline)
        cnt = 1
        for i in range(len(self.story)):
            for pos, word in enumerate(self.outline):
                if i - len(word) + 1 < 0:
                    continue
                if self.order[pos] != 0:
                    continue
                if self.story[i - len(word) + 1: i + 1] == word:
                    self.order[pos] = cnt
                    cnt += 1
        assert cnt == len(self.outline) + 1


class Example(object):
    def __init__(self, outline, order, title):
        self.order = order
        self.outline = outline
        self.title = title


class OrderDataset(torch.utils.data.Dataset):
    def __init__(self, Examples, tokenizer, args):
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
            input = ""
            for idx, word in enumerate(item.outline):
                input += f"<S{idx+1}>" + word
            input += "[EOS]"
            output = "[orig]"
            for idx in item.order:
                output += f"<S{idx}>"
            # output += "[EOS]"
            input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input))
            input_mask = [1] * len(input_ids)
            output_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(output))
            output_mask = [1] * len(output_ids)
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
            "output_mask": []
        }
        for b in batch:
            for k, v in b.items():
                out[k].append(v)
        input_max_pad = 0
        output_max_pad = 0
        if self.config["BUCKET"]:
            for p in out["input_ids"]:
                input_max_pad = max(input_max_pad, len(p))
            for p in out["output_ids"]:
                output_max_pad = max(output_max_pad, len(p))
        else:
            input_max_pad, output_max_pad = self.config["FIX_LENGTH"], self.config["FIX_LENGTH"]
        for i in range(len(batch)):
            input_ids = out["input_ids"][i]
            input_mask = out["input_mask"][i]
            output_ids = out["output_ids"][i]
            output_mask = out["output_mask"][i]
            input_len = len(input_ids)
            output_len = len(output_ids)
            out["input_ids"][i] = input_ids + [self.config["PAD_ID"]] * (input_max_pad - input_len)
            out["input_mask"][i] = input_mask + [0] * (input_max_pad - input_len)
            out["output_ids"][i] = output_ids + [self.config["PAD_ID"]] * (output_max_pad - output_len)
            out["output_mask"][i] = output_mask + [0] * (output_max_pad - output_len)
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


def write_json(data ,path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            out = {
                "outline": item.outline,
                "order": item.order,
                "story": item.story,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def build_order(data):
    raw_examples = []
    logging.info(f"dataset len :{len(data)}")
    for idx, item in enumerate(data):
        # utils.debug("story", item["story"])
        # utils.debug("outline", item["outline"])
        # utils.debug("title", item["title"])
        if idx % 10 == 0:
            logging.info(f"start build {idx}")
        raw_examples.append(Raw_Example(item["story"], item["outline"]))
        raw_examples[-1].build_order()
    return raw_examples


def build_data(args):
    train_data = read_json(args.train_path)
    train_examples = build_order(train_data)
    valid_data = read_json(args.valid_path)
    valid_examples = build_order(valid_data)
    test_data = read_json(args.test_path)
    test_examples = build_order(test_data)
    logging.info(f"train_len:{len(train_examples)}")
    logging.info(f"valid_len:{len(valid_examples)}")
    write_json(train_examples, args.train_save)
    write_json(valid_examples, args.valid_save)
    write_json(test_examples, args.test_save)


def example_data(data, args):
    examples = []
    for item in data:
        examples.append(Example(outline=item["outline"], order=item["order"], title=""))
        args.max_length = max(args.max_length, len(item["order"]))
    return examples


def prepare_data(path, args):
    data = read_json(path)
    examples = example_data(data, args)
    return examples


def extend_Examples(raw_examples):
    new_examples = []
    for item in raw_examples:
        list = [i for i in range(len(item.order))]
        time = 0
        while time < 10:
            new_order = [item.order[i] for i in list]
            new_outline = [item.outline[i] for i in list]
            new_title = item.title
            random.shuffle(list)
            new_examples.append(Example(order=new_order, outline=new_outline, title=new_title))
            time += 1
    return new_examples


def main(args):
    args.max_length = 0
    logging.info("Read Data")
    train_examples = prepare_data(args.train_path, args)
    train_examples = extend_Examples(train_examples)
    valid_examples = prepare_data(args.valid_path, args)
    # test_examples = prepare_data(args.test_path, args)
    args.gold = []
    for item in valid_examples:
        args.gold.append(item.order)
    logging.info("Load Pre-train Model and Tokenizer")
    model = BartForConditionalGeneration.from_pretrained(args.pretrain_path).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    special_token = {"additional_special_tokens":['[shuffle]', '[orig]', '[EOS]', '[title]']}
    extra_token = [f'<S{str(i+1)}>' for i in range(args.max_length)]
    # special_token["additional_special_tokens"].extend(extra_token)
    tokenizer.add_special_tokens(special_token)
    tokenizer.add_tokens(extra_token)
    tokenizer.pad_token = '[PAD]'
    tokenizer.eos_token = '[EOS]'
    tokenizer.bos_token = '[orig]'
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.forced_eos_token_id = tokenizer.eos_token_id
    args.pad_id = tokenizer.pad_token_id
    utils.debug("model.config.eos_token_id", model.config.eos_token_id)
    utils.debug("model.config.bos_token_id", model.config.bos_token_id)
    utils.debug("model.config.decoder_start_token_id", model.config.decoder_start_token_id)
    model.resize_token_embeddings(len(tokenizer))
    logging.info("Prepare Dataset and Iterator")
    train_dataset = OrderDataset(train_examples, tokenizer, args)
    valid_dataset = OrderDataset(valid_examples, tokenizer, args)
    # test_dataset = OrderDataset(test_examples, tokenizer, args)
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=Collection(args))
    valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=Collection(args))
    # test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=Collection(args))
    
    logging.info("Start Training")
    Order_train(train_iter, valid_iter, model, tokenizer, args)


def predict(args):
    args.max_length = 0
    logging.info("Read Data")
    train_examples = prepare_data(args.train_path, args)
    train_examples = extend_Examples(train_examples)
    valid_examples = prepare_data(args.valid_path, args)
    test_examples = prepare_data(args.test_path, args)
    args.gold = []
    for item in valid_examples:
        args.gold.append(item.order)
    logging.info("Load Pre-train Model and Tokenizer")
    model = BartForConditionalGeneration.from_pretrained(args.pretrain_path).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    special_token = {"additional_special_tokens":['[shuffle]', '[orig]', '[EOS]', '[title]']}
    extra_token = [f'<S{str(i+1)}>' for i in range(args.max_length)]
    # special_token["additional_special_tokens"].extend(extra_token)
    tokenizer.add_special_tokens(special_token)
    tokenizer.add_tokens(extra_token)
    tokenizer.pad_token = '[PAD]'
    tokenizer.eos_token = '[EOS]'
    tokenizer.bos_token = '[orig]'
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.forced_eos_token_id = tokenizer.eos_token_id
    args.pad_id = tokenizer.pad_token_id
    utils.debug("model.config.eos_token_id", model.config.eos_token_id)
    utils.debug("model.config.bos_token_id", model.config.bos_token_id)
    utils.debug("model.config.decoder_start_token_id", model.config.decoder_start_token_id)
    model.resize_token_embeddings(len(tokenizer))
    logging.info("Prepare Dataset and Iterator")
    train_dataset = OrderDataset(train_examples, tokenizer, args)
    valid_dataset = OrderDataset(valid_examples, tokenizer, args)
    test_dataset = OrderDataset(test_examples, tokenizer, args)
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=Collection(args))
    valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=Collection(args))
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=Collection(args))
    
    logging.info("Start Training")
    Order_train(train_iter, valid_iter, model, tokenizer, args)


if __name__ == "__main__":
    utils.set_seed(19980917)
    args = Ordering_config()
    if args.build_data:
        build_data(args)
    elif args.train:
        args.model_save = '/'.join([args.model_save, utils.d2s(datetime.datetime.now(), time=True)])
        main(args)
    # elif args.predict:
    #     predict(args)
