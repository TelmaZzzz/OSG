from codecs import encode
import os
import argparse
import logging
import nltk
import random
from utils import *
import pandas as pd
import numpy as np
import json
import re
import rake
import copy
logging.getLogger().setLevel(logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument("--dir_path", type=str, default="/Users/zhanglingyuan/Datasets/Gutenberg/txt")
parser.add_argument("--save_path", type=str, default="/Users/zhanglingyuan/opt/tiger/polish/data/Gutenberg.txt")
parser.add_argument("--rawdata_path", type=str, default="/Users/zhanglingyuan/opt/tiger/polish/data/Gutenberg.txt")
parser.add_argument("--meta_path", type=str)
parser.add_argument("--label", type=str, default="news")
parser.add_argument("--minidata_size", type=int, default=100)
parser.add_argument("--minidata_save_path", type=str, default="/Users/zhanglingyuan/opt/tiger/polish/data")
parser.add_argument("--prepare_rawdata", action="store_true")
parser.add_argument("--prepare_minidata", action="store_true")
parser.add_argument("--cropus_type", type=str, default="Gutenberg")
parser.add_argument("--max_length", type=int, default=50)


def find_position(sentences, position):
    for i in range(max(len(sentences) - position - 1, position)):
        # logging.info(count_word_from_sentence(sentences[position + i]))
        if position + i < len(sentences) - 1 and judge_in(count_word_from_sentence(sentences[position + i]), 20, 35):
            return position + i
        if position - i > 0 and judge_in(count_word_from_sentence(sentences[position - i]), 20, 35):
            return position - i
    return None


def build_minidata(paragraphs):
    minidata = []
    idx = 1
    for paragraph in paragraphs:
        word_list = paragraph.split(" ")
        sentences = []
        sentence = ""
        for word in word_list:
            if judge_sentence(word):
                sentences.append(sentence + word)
            else:
                sentence += " " + word
        if len(sentences) >= 3:
            line = dict()
            line["id"] = idx
            line["paragraph"] = paragraph
            position = (1 + len(sentences)) // 2 -1
            position = find_position(sentences, position)
            if position is None:
                continue
            line["summarization_pos"] = position
            line["summarization_sen"] = sentences[position]
            minidata.append(line)
            idx += 1
    return minidata


def prepare_Brown_rawdata(args):
    meta_data = read_from_csv(args.meta_path)
    filename = []
    for meta in zip(meta_data["filename"], meta_data["label"]):
        if meta[1] == args.label:
            filename.append(meta[0])
    pages = []
    for file in filename:
        logging.info(file)
        txt_data = read_data("/".join([args.dir_path, file]))
        paragraph = []
        for txt in txt_data:
            if txt == "\n":
                if len(paragraph) >= args.max_length:
                    pages.append(paragraph)
                paragraph = []
            else:
                txt = txt.split(" ")
                for word in txt:
                    word_in = word.split("/")[0]
                    if word_in != "\n":
                        paragraph.append(word.split("/")[0])
    with open(args.save_path, "w", encoding="utf-8") as file:
        for page in pages:
            paragraph = ""
            for word in page:
                paragraph += word + " "
            paragraph = paragraph.strip()
            paragraph += "\n"
            file.write(paragraph)


def prepare_Gutenberg_rawdata(args):
    txt_list = os.listdir(args.dir_path)
    pages = []
    for txt in txt_list:
        try:
            pages += read_rawdata('/'.join([args.dir_path, txt]), args.max_length)
        except:
            pass
    with open(args.save_path, "w", encoding="utf-8") as file:
        for page in pages:
            paragraph = ""
            for word in page:
                paragraph += word + ' '
            paragraph += '\n'
            file.write(paragraph)


def prepare_bookcroups_rawdata(args):
    txt_list = os.listdir(args.dir_path)
    pages = []
    logging.info("txt count: {}".format(len(txt_list)))
    with open(args.save_path, "w", encoding="utf-8") as f_out:
        for txt in txt_list:
            logging.info("Start read {}.".format(txt))
            try:
                pages = read_rawdata_new_updn('/'.join([args.dir_path, txt]), args.max_length, args.max_length + 200)
                for page in pages:
                    f_out.write(page)
            except Exception as e:
                logging.warn("e:{}".format(e))


def pos_find(story, pos):
    while pos >=0:
        if story[pos] == "。" or story[pos] == "！" or story[pos] == "”" or story[pos] == "？" \
            or story[pos] == "?" or story[pos] == "!":
            return pos + 1
        pos -= 1


def clean(s):
    s = re.sub(r"\_.*\_", "", s.replace("◇", "_"))
    s = re.sub(r"\(.*\)", "", s.replace("_", ""))
    s = re.sub(r"\[.*\]", "", s)
    s = re.sub(r"（.*）", "", s)
    s = re.sub(r"．．．．．．", "。", s)
    s = re.sub(r"\.\.\.\.\.\.", "", s)
    s = s.replace("\xa0", "").replace("/", "").replace(" ", "")
    s = s.replace("\\", "")
    return s


def prepare_chinese_tonghua_rawdata(args):
    raw_data = read_data(args.rawdata_path)
    data = []
    for item in raw_data:
        item = json.loads(item)
        item["title"] = clean(item["title"])
        item["story"] = clean(item["story"])
        if "作者" in item["story"] or "转载" in item["story"] or "版权" in item["story"] or "www" in item["story"] or "简介" in item["story"]:
            continue
        # if '"' in item["story"] or '”' in item["story"] or '“' in item["story"]:
        #     continue
        if len(item["story"]) <= 90:
            continue
        if len(item["story"]) >= args.max_length:
            pos = pos_find(item["story"], args.max_length - 1)
            item["story"] = item["story"][:pos]
        data.append(item)
    with open(args.save_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False)+"\n")


def prepare_rake_rawdata(args):
    raw_data =read_data(args.rawdata_path)
    data = []
    for item in raw_data:
        try:
            item = json.loads(item)
            story = item["story"]
            result = rake.run(story)
            res = [k for k, _ in result]
            res = res[:min(len(res), 8)]
            item["outline"] = res
            data.append(item)
        except:
            continue
    with open(args.save_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False)+"\n")


def judge_outline_rerake(outline, rerake):
    for word in outline:
        if word.find(rerake) != -1 or rerake.find(word) != -1:
            return False
    return True


def rerake_fn(item, T=3):
    result = rake.run(item["story"])
    re_result = []
    for res in result:
        if judge_outline_rerake(item["outline"], res[0]):
            re_result.append(res[0])
    p = list(range(len(re_result)))
    q = list(range(len(item["outline"])))
    add = []
    while T != 0:
        T -= 1
        random.shuffle(p)
        random.shuffle(q)
        new_item = copy.deepcopy(item)
        new_outline = [item["outline"][i] for i in q[:4]] + [re_result[i] for i in p[:min(len(p),4)]]
        random.shuffle(new_outline)
        new_item["outline"] = new_outline
        add.append(new_item)
    return add


def prepare_rerake_rawdata(args):
    raw_data = read_data(args.rawdata_path)
    data = []
    for item in raw_data:
        # try:
        add = rerake_fn(json.loads(item), T=3)
        data.extend(add)
        # except:
            # add = json.loads(item)
            # data.append(add)
    with open(args.save_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False)+"\n")


def permute_fn(item, T):
    outline = item["outline"]
    idx_list = list(range(len(outline)))
    add = []
    while T != 0:
        T -= 1
        new_item = copy.deepcopy(item)
        new_item["outline"] = [outline[idx] for idx in idx_list]
        add.append(new_item)
        random.shuffle(idx_list)
    return add


def prepare_permute_rawdata(args):
    raw_data = read_data(args.rawdata_path)
    data = []
    for item in raw_data:
        add = permute_fn(json.loads(item), T=6)
        data.extend(add)
    with open(args.save_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False)+"\n")


def prepare_rawdata(args):
    if args.cropus_type == "Brown":
        prepare_Brown_rawdata(args)
    elif args.cropus_type == "Gutenberg":
        prepare_Gutenberg_rawdata(args)
    elif args.cropus_type == "bookcroups":
        prepare_bookcroups_rawdata(args)
    elif args.cropus_type == "chinese_tonghua":
        prepare_chinese_tonghua_rawdata(args)
    elif args.cropus_type == "rake":
        prepare_rake_rawdata(args)
    elif args.cropus_type == "rerake":
        prepare_rerake_rawdata(args)
    elif args.cropus_type == "permute":
        prepare_permute_rawdata(args)


def prepare_minidata(args):
    prefix = args.rawdata_path.split("/")[-1].split(".")[0]
    paragraph = read_data(args.rawdata_path)
    random.shuffle(paragraph)
    minidata = build_minidata(paragraph[:min(len(paragraph), args.minidata_size)])
    path = "/".join([args.minidata_save_path, prefix + ".json"])
    with open(path, "w", encoding="utf-8") as file:
        for line in minidata:
            json.dump(line, file)
            file.write("\n")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.prepare_rawdata:
        logging.info("Start prepare rawdata")
        prepare_rawdata(args)
    elif args.prepare_minidata:
        logging.info("Start prepare minidata")
        prepare_minidata(args)
    logging.info("END...")
