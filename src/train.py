import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup, AdamW, get_cosine_with_hard_restarts_schedule_with_warmup
import logging
from utils import utils
import metrics
import math
import json
import torch.distributed as dist


def save(model, path, step):
    # if dist.get_rank() == 0:
    path += "_epoch{}.pkl".format("{}".format(step))
    # if os.path.exists(path):
    #     os.remove(path)
    #     logging.info("model remove success!!!")
    logging.info("Save model")
    torch.save(model.module, path)


def draw(f, mp):
    f.write(json.dumps(mp, ensure_ascii=False)+"\n")
    # f.write("outline: " + utils.list2str(outline) + "\n")
    # f.write("gold:\n")
    # f.write(gold+"\n")
    # f.write("predict:\n")
    # f.write(predict +"\n")
    # f.write("-----------------------------------------------\n")


def ComGen_valid(valid_iter, model, tokenizer, args):
    logging.info("Start valid")
    model.eval()
    predicts = []
    for item in valid_iter:
        logit = model.generate(item["input_ids"].cuda(), num_beams=16, max_length=20, early_stopping=True)
        # logit = model(input_ids=item["input_ids"].cuda(), attention_mask=item["attention_mask"].cuda()).logits
        # logit = torch.max(F.softmax(logit, dim=-1), dim=-1)[1].cpu()
        predict = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in logit]
        predicts.extend(predict)
    save(model, args.model_save, args.step)
    with open(args.model_save + "_epoch{}.txt".format(args.step), "w", encoding="utf-8") as f:
        for real, predict in zip(args.reals, predicts):
            f.write("real: " + real + "\n")
            f.write("predict: " + predict + "\n")
            f.write("----------------------------------------------\n")
            f.write("\n")


def ComGen_train(train_iter, valid_iter, model, tokenizer, args):
    optimizer = AdamW(model.parameters(), args.learning_rate, weight_decay=0.0001, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_iter), num_training_steps=len(train_iter) * args.epoch)
    mean_loss = 0
    for step in range(args.epoch):
        model.train()
        logging.info("Starting Training epoch:{}".format(step+1))
        for item in train_iter:
            loss = model(input_ids=item["input_ids"].cuda(), attention_mask=item["attention_mask"].cuda(), labels=item["label"].cuda()).loss
            mean_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        args.step = step + 1
        mean_loss /= len(train_iter)
        logging.info("Train loss:{:.4f}".format(mean_loss))
        mean_loss = 0
        ComGen_valid(valid_iter, model, tokenizer, args)


def Order_valid(valid_iter, model, tokenizer, args):
    model.eval()
    predicts = []
    for item in valid_iter:
        input_ids = item["input_ids"]
        # attention_mask = item["input_mask"]
        # output_ids = item["output_ids"]
        # output_mask = item["output_mask"]
        # decoder_input_ids = output_ids[:, 0:1].contiguous()
        logits = model.generate(input_ids=input_ids.cuda(), no_repeat_ngram_size=1, num_beams=10, max_length=args.max_length + 2)
        predict = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in logits]
        predicts.extend(predict)
    acc, PMR, kendall_score, LCS, rouge = 0, 0, 0, 0, 0
    total, total_sents = 0, 0
    with open(args.model_save + f"_epoch{args.step}.txt", "w", encoding="utf-8") as f:
        for gold, predict in zip(args.gold, predicts):
            predict = predict.replace("<S", "").replace(">", "").split(" ")
            predict = [int(item) for item in predict]
            f.write(utils.list2str(predict)+"\n")
            total += 1
            total_sents += len(gold)
            acc += metrics.acc_compare(gold, predict)
            ktua = metrics.kendall_tau(predict, gold)
            LCS += metrics.lcs(gold, predict)
            rouge += metrics.rouge_s(gold, predict)
            if gold == predict:
                PMR += 1
            if math.isnan(ktua):
                ktua = 0
            kendall_score += ktua
        logging.info(" Accuracy: {:.6f}".format(acc / total_sents))
        logging.info(" PMR: {:.6f}".format(PMR / total))
        logging.info(" Kendall's Tau: {:.6f}".format(kendall_score / total))
        logging.info(" LCS: {:.6f}".format(LCS / total_sents))
        logging.info(" Rouge-S: {:.6f}".format(rouge / total))
    save(model, args.model_save, args.step)
        # utils.debug("predict", predict)
        # utils.debug("gold", gold)
    # utils.debug("predict:", predict)
        # optimizer.zero_grad()
        # lm_logits = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, use_cache=False).logits
        # batch_size, length, vocab_size = lm_logits.shape
        # lm_token_ids = torch.max(F.softmax(lm_logits, dim=-1), dim=-1)[1]


def Order_train(train_iter, valid_iter, model, tokenizer, args):
    utils.debug("model", model)
    no_decay = ["bias", "LayerNorm.weight"]
    high_lr = ["lm_head"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and not any(nd in n for nd in high_lr)
            ],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate * 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": args.learning_rate * 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in high_lr)
            ],
            "weight_decay": args.weight_decay,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_iter), num_training_steps=len(train_iter) * args.epoch)
    for step in range(args.epoch):
        args.step = step
        model.train()
        loss_mean = 0
        for item in train_iter:
            input_ids = item["input_ids"]
            attention_mask = item["input_mask"]
            output_ids = item["output_ids"]
            output_mask = item["output_mask"]
            decoder_input_ids = output_ids[:, :-1].contiguous()
            lm_logits = model(input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda(), decoder_input_ids=decoder_input_ids.cuda(), use_cache=False).logits.cpu()
            batch_size, length, vocab_size = lm_logits.shape
            Loss_fn = nn.CrossEntropyLoss(reduction="none")
            lm_labels = output_ids[:, 1:].clone().contiguous()
            # utils.debug("lm_label", lm_labels.shape)
            # utils.debug("lm_logit", lm_logits.shape)
            loss = Loss_fn(lm_logits.view(-1, vocab_size), lm_labels.view(-1)).view(batch_size, length)
            loss_mask = output_mask[:, :-1].contiguous()
            loss = torch.mul(loss_mask, loss)
            # utils.debug("loss", loss.shape)
            loss = torch.mean(loss)
            loss_mean += loss.item()
            # utils.debug("loss", loss.shape)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        logging.info(f"epoch:{step+1} loss:{loss_mean / len(train_iter)}")
        loss_meam = 0
        with torch.no_grad():
            Order_valid(valid_iter, model, tokenizer, args)


def Base_predict(test_iter, model, tokenizer, args):
    model.eval()
    predict_logits = []
    predicts = []
    parameter = args.parameter
    for item in test_iter:
        # if args.n_gpu > 1:
            # logit = model.module.generate(item["input_ids"].to(args.device), do_sample=False, top_p=0.9, min_length=200, max_length=512)
        # else:
            # logit = model.generate(item["input_ids"].to(args.device), do_sample=False, top_p=0.9, min_length=200, max_length=512)
        if args.n_gpu > 1:
            logit = model.module.generate(item["input_ids"].to(args.device), max_length=parameter["max_length"], \
                min_length=parameter["min_length"], do_sample=parameter["do_sample"], early_stopping=parameter["early_stopping"], \
                num_beams=parameter["num_beams"], temperature=parameter["temperature"], top_k=parameter["top_k"], top_p=parameter["top_p"], \
                length_penalty=parameter["length_penalty"], no_repeat_ngram_size=parameter["no_repeat_ngram_size"])
        else:
            logit = model.module.generate(item["input_ids"].to(args.device), max_length=parameter["max_length"], \
                min_length=parameter["min_length"], do_sample=parameter["do_sample"], early_stopping=parameter["early_stopping"], \
                num_beams=parameter["num_beams"], temperature=parameter["temperature"], top_k=parameter["top_k"], top_p=parameter["top_p"], \
                length_penalty=parameter["length_penalty"])
        # predict = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False).replace(" ","").replace("[unused1]", "") for g in logit]
        # utils.debug("predict", predict[0])
        batch_size, seq_len = logit.shape
        pad = torch.tensor([0] * (batch_size * (512-seq_len)), dtype=torch.long).reshape(batch_size, -1).to(args.device)
        predict_logits.append(torch.cat([logit, pad], dim=-1))
        # predicts.extend(predict)
    predictions = utils.distributed_concat(torch.cat(predict_logits, dim=0), args.test_len)
    predicts = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False).replace(" ","").replace("[unused1]", "") for g in predictions]
    logging.info(f"predicts len: {len(predicts)}")
    if dist.get_rank() != 0:
        dist.barrier()
    else:
        logging.info("Metrics Compare")
        res = metrics.base_compare(args.gold, predicts, args.outline)
        overall = metrics.overall_compare(res)
        res["overall"] = overall
        name = parameter["name"]
        logging.info(f"generate parameter name: {name}")
        for k, v in res.items():
            logging.info("{}: {:.4f}".format(k, v))
        with open(args.output + f"_{name}.txt", "w", encoding="utf-8") as f:
            for i in range(len(predicts)):
                f.write(predicts[i].strip()+"\n")
        with open(args.ans_list, "a", encoding="utf-8") as f:
            # for i in range(len(predicts)):
            #     mp = {
            #         "outline": args.outline[i],
            #         "gold": args.gold[i],
            #         "predicts": predicts[i]
            #     }
            #     draw(f, mp)
                # f.write("outline: " + utils.list2str(args.outline[i]) + "\n")
                # f.write("gold:\n")
                # f.write(args.gold[i]+"\n")
                # f.write("predict:\n")
                # f.write(predicts[i]+"\n")
                # f.write("-----------------------------------------------\n")
            res["path"] = args.output + f"_{name}.txt"
            draw(f, res)
        dist.barrier()


def Base_valid(valid_iter, model, tokenizer, args):
    model.eval()
    predict_logits = []
    parameter = args.parameter
    for item in valid_iter:
        if args.n_gpu > 1:
            logit = model.module.generate(item["input_ids"].to(args.device), max_length=parameter["max_length"], \
                min_length=parameter["min_length"], do_sample=parameter["do_sample"], early_stopping=parameter["early_stopping"], \
                num_beams=parameter["num_beams"], temperature=parameter["temperature"], top_k=parameter["top_k"], top_p=parameter["top_p"], \
                length_penalty=parameter["length_penalty"], no_repeat_ngram_size=parameter["no_repeat_ngram_size"])
        else:
            logit = model.module.generate(item["input_ids"].to(args.device), max_length=parameter["max_length"], \
                min_length=parameter["min_length"], do_sample=parameter["do_sample"], early_stopping=parameter["early_stopping"], \
                num_beams=parameter["num_beams"], temperature=parameter["temperature"], top_k=parameter["top_k"], top_p=parameter["top_p"], \
                length_penalty=parameter["length_penalty"])
        # logit = model(input_ids=item["input_ids"].cuda(), attention_mask=item["attention_mask"].cuda()).logits
        # logit = torch.max(F.softmax(logit, dim=-1), dim=-1)[1].cpu()
        # utils.debug("logit shape", logit.shape)
        batch_size, seq_len = logit.shape
        pad = torch.tensor([0] * (batch_size * (512-seq_len)), dtype=torch.long).reshape(batch_size, -1).to(args.device)
        predict_logits.append(torch.cat([logit, pad], dim=-1))
        # predict = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False).replace(" ","").replace("[unused1]", "") for g in logit]
        # utils.debug("predict", predict[0])
        # predicts.extend(predict)
    if args.local_rank != -1:
        predictions = utils.distributed_concat(torch.cat(predict_logits, dim=0), args.valid_len)
    else:
        predictions = torch.cat(predict_logits, dim=0)
    predicts = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False).replace(" ","").replace("[unused1]", "") for g in predictions]
    if dist.get_rank() != 0:
        dist.barrier()
    else:
        logging.info("Metrics Compare")
        res = metrics.base_compare(args.gold, predicts, args.outline)
        overall = metrics.overall_compare(res)
        res["overall"] = overall
        for k, v in res.items():
            logging.info("{}: {:.4f}".format(k, v))
        save(model, args.model_save, args.step)
        with open(args.model_save + f"_epoch{args.step}.txt", "w", encoding="utf-8") as f:
            for i in range(len(predicts)):
                f.write(predicts[i].strip()+"\n")
        with open(args.model_save + f"_epoch{args.step}.jsonl", "w", encoding="utf-8") as f:
            for i in range(len(predicts)):
                mp = {
                    "outline": args.outline[i],
                    "gold": args.gold[i],
                    "predicts": predicts[i]
                }
                draw(f, mp)
                # f.write("outline: " + utils.list2str(args.outline[i]) + "\n")
                # f.write("gold:\n")
                # f.write(args.gold[i]+"\n")
                # f.write("predict:\n")
                # f.write(predicts[i]+"\n")
                # f.write("-----------------------------------------------\n")
            draw(f, res)
        dist.barrier()


def Base_train(train_iter, valid_iter, model, tokenizer, args):
    no_decay = ["bias", "LayerNorm.weight"]
    high_lr = ["lm_head"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and not any(nd in n for nd in high_lr)
            ],
            "weight_decay": args.weight_decay,
            # "lr": args.learning_rate * 0.1,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            # "lr": args.learning_rate * 0.1,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in high_lr)
            ],
            "weight_decay": args.weight_decay,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, args.learning_rate, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_iter) // args.opt_step, num_training_steps=len(train_iter) * args.epoch // args.opt_step)
    # scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=len(train_iter) // args.opt_step, num_training_steps=len(train_iter) * args.epoch // args.opt_step)
    mean_loss = 0
    for step in range(args.epoch):
        if args.local_rank != -1:
            train_iter.sampler.set_epoch(step)
        model.train()
        logging.info("Starting Training epoch:{}".format(step+1))
        for idx, item in enumerate(train_iter):
            loss = model(input_ids=item["input_ids"].to(args.device), attention_mask=item["input_mask"].to(args.device), labels=item["output_ids"].to(args.device)).loss
            if args.n_gpu > 1:
                loss = torch.mean(loss)
            loss.backward()
            mean_loss += loss.cpu().item()
            if idx % args.opt_step == args.opt_step - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        args.step = step + 1
        mean_loss /= len(train_iter)
        logging.info("Train loss:{:.4f}".format(mean_loss))
        mean_loss = 0
        # if dist.get_rank() == 0:
        if step < 5:
            continue
        with torch.no_grad():
            Base_valid(valid_iter, model, tokenizer, args)


def Rewrite_valid(valid_iter, model, tokenizer, args):
    model.eval()
    predicts = []
    for item in valid_iter:
        if args.n_gpu > 1:
            logit = model.module.generate(item["input_ids"].cuda(), num_beams=15, min_length=20, max_length=100, early_stopping=True, no_repeat_ngram_size=4)
        else:
            logit = model.generate(item["input_ids"].cuda(), num_beams=15, min_length=20, max_length=100, early_stopping=True, no_repeat_ngram_size=4)
        # logit = model(input_ids=item["input_ids"].cuda(), attention_mask=item["attention_mask"].cuda()).logits
        # logit = torch.max(F.softmax(logit, dim=-1), dim=-1)[1].cpu()
        predict = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False).replace(" ","").replace("[unused1]", "") for g in logit]
        # utils.debug("predict", predict[0])
        predicts.extend(predict)
    logging.info("Metrics Compare")
    res = metrics.base_compare(args.gold, predicts, args.outline)
    overall = metrics.overall_compare(res)
    res["overall"] = overall
    for k, v in res.items():
        logging.info("{}: {:.4f}".format(k, v))
    save(model, args.model_save, args.step)
    with open(args.model_save + f"_epoch{args.step}.txt", "w", encoding="utf-8") as f:
        for i in range(len(predicts)):
            f.write("outline: " + utils.list2str(args.outline[i]) + "\n")
            f.write("gold:\n")
            f.write(args.gold[i]+"\n")
            f.write("predict:\n")
            f.write(predicts[i]+"\n")
            f.write("-----------------------------------------------\n")
        for k, v in res.items():
            f.write("{} : {:.4f}\n".format(k, v))


def Rewrite_train(train_iter, valid_iter, model, tokenizer, args):
    no_decay = ["bias", "LayerNorm.weight"]
    high_lr = ["lm_head"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and not any(nd in n for nd in high_lr)
            ],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate * 0.1,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": args.learning_rate * 0.1,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in high_lr)
            ],
            "weight_decay": args.weight_decay,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, args.learning_rate, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_iter), num_training_steps=len(train_iter) * args.epoch)
    mean_loss = 0
    for step in range(args.epoch):
        model.train()
        logging.info("Starting Training epoch:{}".format(step+1))
        for item in train_iter:
            utils.debug("input_ids shape", item["input_ids"].shape)
            utils.debug("output_ids shape", item["output_ids"].shape)
            # utils.debug("input_ids", item["input_ids"])
            utils.debug("input_mask shape", item["input_mask"].shape)
            # utils.debug("output_ids", item["output_ids"])
            try:
                loss = model(input_ids=item["input_ids"].cuda(), attention_mask=item["input_mask"].cuda(), labels=item["output_ids"].cuda()).loss
                optimizer.zero_grad()
                if args.n_gpu > 1:
                    loss = torch.mean(loss)
                loss.backward()
                mean_loss += loss.cpu().item()
                optimizer.step()
                scheduler.step()
            except Exception as e:
                logging.warning(f"error msg: {e}")
        args.step = step + 1
        mean_loss /= len(train_iter)
        logging.info("Train loss:{:.4f}".format(mean_loss))
        mean_loss = 0
        Rewrite_valid(valid_iter, model, tokenizer, args)


def Rewrite_predict(predict_iter, model, tokenizer, args):
    model.eval()
    predicts = []
    for item in predict_iter:
        if args.n_gpu > 1:
            logit = model.module.generate(item["input_ids"].cuda(), num_beams=15, min_length=20, max_length=100, early_stopping=True, no_repeat_ngram_size=4)
        else:
            logit = model.generate(item["input_ids"].cuda(), num_beams=15, min_length=20, max_length=100, early_stopping=True, no_repeat_ngram_size=4)
        # logit = model(input_ids=item["input_ids"].cuda(), attention_mask=item["attention_mask"].cuda()).logits
        # logit = torch.max(F.softmax(logit, dim=-1), dim=-1)[1].cpu()
        predict = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False).replace(" ","").replace("[unused1]", "") for g in logit]
        # utils.debug("predict", predict[0])
        predicts.extend(predict)
    # save(model, args.model_save, args.step)
    new_predict = []
    new_gold = []
    new_outline = []
    with open(args.rewrite_save, "w", encoding="utf-8") as f:
        for i in range(len(predicts)):
            if args.error_data.get(i, None) is not None:
                for item in args.error_data[i]:
                    draw(f, item.outline, item.gold_story, item.predict_story)
                    new_outline.append(item.outline)
                    new_gold.append(item.gold_story)
                    new_predict.append(item.predict_story)
            draw(f, args.gold[i].old_outline, args.gold[i].target, args.gold[i].up_content + predicts[i] + args.gold[i].dn_content)
            new_outline.append(args.gold[i].old_outline)
            new_gold.append(args.gold[i].target)
            new_predict.append(args.gold[i].up_content + predicts[i] + args.gold[i].dn_content)
    logging.info("Metrics Compare")
    res = metrics.base_compare(new_gold, new_predict, new_outline)
    overall = metrics.overall_compare(res)
    res["overall"] = overall
    for k, v in res.items():
        logging.info("{}: {:.4f}".format(k, v))
    with open(args.rewrite_save, "a", encoding="utf-8") as f:
        for k, v in res.items():
            f.write("{} : {:.4f}\n".format(k, v))


def OrderBase_valid(valid_iter, model, tokenizer, args):
    model.eval()
    # predicts = []
    predict_logits = []
    parameter = args.parameter
    for item in valid_iter:
        if args.n_gpu > 1:
            logit = model.module.generate(item["input_ids"].to(args.device), max_length=parameter["max_length"], \
                min_length=parameter["min_length"], do_sample=parameter["do_sample"], early_stopping=parameter["early_stopping"], \
                num_beams=parameter["num_beams"], temperature=parameter["temperature"], top_k=parameter["top_k"], top_p=parameter["top_p"], \
                length_penalty=parameter["length_penalty"], no_repeat_ngram_size=parameter["no_repeat_ngram_size"])
        else:
            logit = model.module.generate(item["input_ids"].to(args.device), max_length=parameter["max_length"], \
                min_length=parameter["min_length"], do_sample=parameter["do_sample"], early_stopping=parameter["early_stopping"], \
                num_beams=parameter["num_beams"], temperature=parameter["temperature"], top_k=parameter["top_k"], top_p=parameter["top_p"], \
                length_penalty=parameter["length_penalty"])
        batch_size, seq_len = logit.shape
        pad = torch.tensor([0] * (batch_size * (512-seq_len)), dtype=torch.long).reshape(batch_size, -1).to(args.device)
        predict_logits.append(torch.cat([logit, pad], dim=-1))
        # predict = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False).replace(" ","").replace("[unused1]", "") for g in logit]
        # utils.debug("predict", predict[0])
        # predicts.extend(predict)
    predictions = utils.distributed_concat(torch.cat(predict_logits, dim=0), args.valid_len)
    predicts = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False).replace(" ","").replace("[unused1]", "") for g in predictions]
    if dist.get_rank() != 0:
        dist.barrier()
    else:
        logging.info("Metrics Compare")
        res = metrics.base_compare(args.gold, predicts, args.outline)
        overall = metrics.overall_compare(res)
        res["overall"] = overall
        for k, v in res.items():
            logging.info("{}: {:.4f}".format(k, v))
        save(model, args.model_save, args.step)
        with open(args.model_save + f"_epoch{args.step}.txt", "w", encoding="utf-8") as f:
            for i in range(len(predicts)):
                f.write("outline: " + utils.list2str(args.outline[i]) + "\n")
                f.write("gold:\n")
                f.write(args.gold[i]+"\n")
                f.write("predict:\n")
                f.write(predicts[i]+"\n")
                f.write("-----------------------------------------------\n")
            for k, v in res.items():
                f.write("{} : {:.4f}\n".format(k, v))
        dist.barrier()


def OrderBase_train(train_iter, valid_iter, model, tokenizer, args):
    no_decay = ["bias", "LayerNorm.weight"]
    high_lr = ["lm_head"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and not any(nd in n for nd in high_lr)
            ],
            "weight_decay": args.weight_decay,
            # "lr": args.learning_rate * 0.1,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            # "lr": args.learning_rate * 0.1,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in high_lr)
            ],
            "weight_decay": args.weight_decay,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, args.learning_rate, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_iter) // args.opt_step, num_training_steps=len(train_iter) * args.epoch // args.opt_step)
    # scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=len(train_iter) // args.opt_step, num_training_steps=len(train_iter) * args.epoch // args.opt_step)
    mean_loss = 0
    mean_decoder_loss = 0
    mean_encoder_loss = 0
    pices = args.encoder_loss_p / args.epoch * 5
    for step in range(args.epoch):
        train_iter.sampler.set_epoch(step)
        model.train()
        logging.info("Starting Training epoch:{}".format(step+1))
        for idx, item in enumerate(train_iter):
            output = model(input_ids=item["input_ids"].to(args.device), attention_mask=item["input_mask"].to(args.device), labels=item["output_ids"].to(args.device), \
                encoder_labels=item["encoder_labels"].to(args.device), encoder_labels_idx=item["encoder_labels_idx"])
            loss = output.loss
            encoder_loss = output.encoder_loss
            utils.debug("loss", loss)
            utils.debug("encoder_loss", encoder_loss)
            if args.n_gpu > 1:
                loss = torch.mean(loss)
                encoder_loss = torch.mean(encoder_loss)
                mean_decoder_loss += loss.cpu().item()
                mean_encoder_loss += encoder_loss.cpu().item()
            loss = loss + max(args.encoder_loss_p - step * pices, 0) * encoder_loss
            loss.backward()
            mean_loss += loss.cpu().item()
            if idx % args.opt_step == args.opt_step - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        args.step = step + 1
        mean_loss /= len(train_iter)
        mean_decoder_loss /= len(train_iter)
        mean_encoder_loss /= len(train_iter)
        logging.info("mean_decoder_loss: {:.4f}; mean_encoder_loss: {:.4f}".format(mean_decoder_loss, mean_encoder_loss))
        logging.info("Train loss:{:.4f}".format(mean_loss))
        mean_loss = 0
        mean_decoder_loss, mean_encoder_loss = 0, 0
        # if step < 5:
            # continue
        # if dist.get_rank() == 0:
        with torch.no_grad():
            OrderBase_valid(valid_iter, model, tokenizer, args)
