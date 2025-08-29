# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torch.nn import CosineSimilarity
from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.model_baseline import VQAModel
# from tasks.model import VQAModel
import os
import psutil
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import ot
process = psutil.Process(os.getpid())

import torch.nn.functional as F
import json
import pickle as cPickle  # python3
import utils
import datetime
import  numpy as np


@torch.no_grad()
def evaluate(model, dataloader, qid2type): #self.model, val_loader, qid2type, opt
    score = 0
    upper_bound = 0
    num_data = 0
    entropy = 0

    score_yesno = 0
    score_number = 0
    score_other = 0
    total_yesno = 0
    total_number = 0
    total_other = 0

    for i, (v, b, q, a, q_id) in enumerate(dataloader):
        v = v.cuda()
        b = b.cuda()
        q = list(q)
        q_id = q_id.cuda()
        out_dict = model(v, b, q, False)
        pred = out_dict['logits']
        batch_score = compute_score_with_logits(pred, a.cuda()).cpu().numpy().sum(1)
        score += batch_score.sum()
        upper_bound += (a.max(1)[0]).sum().item()
        num_data += pred.size(0)

        qids = q_id.detach().cpu().int().numpy()
        for j in range(len(qids)):
            qid = qids[j]
            typ = qid2type[str(qid)]
            if typ == 'yes/no':
                score_yesno += batch_score[j]
                total_yesno += 1
            elif typ == 'other':
                score_other += batch_score[j]
                total_other += 1
            elif typ == 'number':
                score_number += batch_score[j]
                total_number += 1
            else:
                print('Hahahahahahahahahahaha')

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    score_yesno /= total_yesno
    score_other /= total_other
    score_number /= total_number

    results = dict(
        score=score,
        upper_bound=upper_bound,
        entropy=entropy,
        score_yesno=score_yesno,
        score_other=score_other,
        score_number=score_number,
    )

    return results

# multi-label soft loss
def instance_bce_with_logits(logits, labels, reduction='mean'):
    assert logits.dim() == 2
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction=reduction)
    if reduction == 'mean':
        loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores



def get_our_data():
    from dataset_tpcl import Dictionary, VQAFeatureDataset
    import utils_1
    from src.param import args as opt

    dictionary = Dictionary.load_from_file(os.path.join(opt.dataroot, 'dictionary.pkl'))
    opt.ntokens = dictionary.ntoken


    # load curriculum
    if opt.cl == "ling_diff_type":
        selected_cl = json.load(open('../cl/cl_order.json'))['curriculum']['linguistic']
    else:
        selected_cl = json.load(open('../cl/cl_order.json'))['curriculum'][opt.cl]


    print("Building train dataset...")
    VQAFeats_lst = []
    for qtype in selected_cl:
        print("qtype: ", qtype)
        clss_obj = VQAFeatureDataset('train', dictionary, opt.dataroot, opt.img_root, ratio=opt.ratio, qtype=qtype,adaptive=False, dataset=opt.dataset)  # , negs_num = opt.negs_num)  # load labeld data
        VQAFeats_lst.append(clss_obj)

    train_qtype_lst = []
    for VQA_obj in VQAFeats_lst:
        train_qtype_lst.append(DataLoader(VQA_obj, opt.batch_size, shuffle=True, num_workers=4, collate_fn=utils_1.trim_collate))
    
    print("Building Test dataset loader...")
    eval_dset = VQAFeatureDataset('test', dictionary, opt.dataroot, opt.img_root,ratio=1.0, adaptive=False, dataset=opt.dataset)
    opt.use_all = 1
    val_loader = DataLoader(eval_dset, opt.batch_size, shuffle=False, num_workers=4, collate_fn=utils_1.trim_collate)

    return train_qtype_lst, val_loader


def evaluate_train(model, dataloader, args):
    score = 0
    upper_bound = 0
    scores_lst = []
    total_loss_lst = []
    ids_lst = []

    # (feats, boxes, sent, sent_pos, target, ques_id)
    for i, (v, b, q, q_pos, a, q_id) in enumerate(dataloader):
        v = v.cuda()
        b = b.cuda()
        q = list(q)
        q_id = q_id.cuda()
        a = a.cuda()
        out_dict = model(v, b, q, False)
        bce_loss = instance_bce_with_logits(out_dict['logits'], a, reduction='none')
        pred = out_dict['logits']
        batch_score = compute_score_with_logits(pred, a.cuda()).cpu().numpy().sum(1)
        score += batch_score.sum()
        upper_bound += (a.max(1)[0]).sum().item()
        scores_lst.extend(batch_score.tolist())  # add scores to list
        total_loss_lst.extend(bce_loss.sum(dim=1).tolist())
        ids_lst.extend(q_id.tolist())

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)

    results = dict(
            score=score,
            upper_bound=upper_bound,
            scores_lst=scores_lst,
            ids_lst=ids_lst,
            total_loss_lst=total_loss_lst,
            )

    return results

def get_curriculum_schedule(train_qtype_lst, model, comp, opt):
    dict_q = {}

    if model is not None:
        model.eval()

        for loader in train_qtype_lst:
            train_results = evaluate_train(model, loader, None, opt)

            dict_q[loader.dataset.qtype] = 1- train_results['score']
        if args.cl_reverse:
            qtype_score_dict = dict(filter(lambda item: item[1] >= comp, dict_q.items()))
        else:
            qtype_score_dict = dict(filter(lambda item: item[1] <= comp, dict_q.items()))
    else:
        qtype_score_dict = json.load(open('/scratch3/akl002/VQA/dvqa39/LXMERT/lxmert_qtype_score_dict.json'))
        if args.cl_reverse:
            qtype_score_dict = dict(filter(lambda item: 1 - item[1] >= comp, qtype_score_dict.items()))
        else:
            qtype_score_dict = dict(filter(lambda item: 1 - item[1] <= comp, qtype_score_dict.items()))

    cl_list = []
    for k, v in qtype_score_dict.items():
        for item in train_qtype_lst:
            if item.dataset.qtype == k:
                cl_list.append(item)

    if model is not None:
        model.train(True)

    if comp == 0:
        current_cl_epoch = 5
    else:
        current_cl_epoch = 3


    return cl_list, current_cl_epoch, qtype_score_dict

def get_cl_schedule(train_loader_lst, comp, qtype_dist_diff_dict, opt):
    qtype_lst  = []

    if opt.cl_reverse:
        qtype_dist_diff_dict_sorted = dict(sorted(qtype_dist_diff_dict.items(), key=lambda item: item[1], reverse=True))
    else:
        qtype_dist_diff_dict_sorted = dict(sorted(qtype_dist_diff_dict.items(), key=lambda item: item[1]))

    qtype_numb = int(comp * len(qtype_dist_diff_dict_sorted))
    cl_list = []
    for k, v in list(qtype_dist_diff_dict_sorted.items())[:qtype_numb+1]:
        qtype_lst.append(k)
    # select the qtype training loaders
    for qtype in qtype_lst:
        for item in train_loader_lst:
            if item.dataset.qtype == qtype:
                cl_list.append(item)
    current_cl_epoch = 5
    return cl_list, current_cl_epoch

count = 0
def compute_dist_diff(data_dict, qtype_dist_diff_dict, opt, warmp_up=True):

    weights = [0.1, 0.1, 0.3, 0.5]
    global count
    json.dump(data_dict, open(os.path.join(opt.output, str(count) + "_qtypes_losses_lst_dict.json"), 'w'))
    count += 1
    for qtype, scores_lst in data_dict.items(): # iterate over qtype scores dict.
        qtype_dist_diff_lst = []
        for i in range(len(scores_lst)-1):
            upper_bound = 100
            max_val = int(max(scores_lst[i]))
            if max_val > 100:
                upper_bound = max_val

            counts, support, _ = plt.hist(scores_lst[i], 101, (0, upper_bound))
            counts2, support2, _ = plt.hist(scores_lst[i+1], 101, (0, upper_bound))
            supports = (support[:-1] + support[1:]) / 2
            d = euclidean_distances(np.array([supports, supports]).T)
            prob1 = counts / np.sum(counts)
            prob2 = counts2 / np.sum(counts2)
            gamma, r = ot.emd(prob1, prob2, d, log=True)
            qtype_dist_diff_lst.append(r['cost'])
        if warmp_up:
            qtype_dist_diff_dict[qtype] = sum(np.multiply(qtype_dist_diff_lst, weights))
        else:
            qtype_dist_diff_dict[qtype] = sum(np.multiply(qtype_dist_diff_lst, weights))
            # qtype_dist_diff_dict[qtype] = sum(qtype_dist_diff_lst)

    return qtype_dist_diff_dict

def warmup_model(model, train_loader_lst, optim, opt, qtype_dist_diff_dict, logger):
    qtype_lst = [train_loader.dataset.qtype for train_loader in train_loader_lst]
    qtype_score_lst_dict = {el: [] for el in qtype_lst}
    qtype_ids_lst_dict = {el: [] for el in qtype_lst}
    qtype_loss_lst_dict = {el: [] for el in qtype_lst}

    N = 0
    for train_loader in train_loader_lst:
        N += len(train_loader.dataset)

    for epoch in range(opt.warmup_epochs):
        train_score = 0
        total_loss = 0
        total_bce_loss = 0
        total_con_loss = 0
        train_score_pos = 0
        for train_loader in train_loader_lst:
            # features, torch.FloatTensor([0]), question, question_pos, target, question_id
            for i, (feats, boxes, sent, target, ques_id) in tqdm(enumerate(train_loader)):
                model.train()
                optim.zero_grad()
                batch_size = feats.size(0)
                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                if args.mode == 'lxmert':
                    out_dict = model(feats, boxes, list(sent))
                    # base VQA model
                    bce_loss = instance_bce_with_logits(out_dict['logits'], target, reduction='mean')
                    loss = bce_loss
            total_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optim.step()
            score_pos = compute_score_with_logits(out_dict['logits'], target.data).sum()
            train_score_pos += score_pos.item()
            total_loss += loss.item() * batch_size
            total_bce_loss += bce_loss.item() * batch_size
        total_loss /= N
        total_bce_loss /= N
        train_score = 100 * train_score / N
        model.train(False)
        # evaluate the model on the training data after the epoch end
        for train_loader in train_loader_lst:
            train_results = evaluate_train(model, train_loader, args)
            qtype_score_lst_dict[train_loader.dataset.qtype].append(train_results['scores_lst'])
            qtype_ids_lst_dict[train_loader.dataset.qtype].append(train_results['ids_lst'])
            qtype_loss_lst_dict[train_loader.dataset.qtype].append(train_results['total_loss_lst'])
        model.train(True)
        logger.write("Warming up Epoch:  " + str(epoch))

    qtype_dist_diff_dict = compute_dist_diff(qtype_loss_lst_dict, qtype_dist_diff_dict, opt, warmp_up=True)

    return model, optim, qtype_dist_diff_dict

class VQA:
    def __init__(self,folder="/",load=True):
        # Datasets
        self.train_loader_lst, self.val_loader = get_our_data()
        self.model = VQAModel(2274) # cpv2 num_ans = 2274, v2 num_ans = 2410

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)
        
        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Loss and Optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        if load :
            if 'bert' in args.optim:
                batch_per_epoch = 0
                for train_loader in self.train_loader_lst:
                    batch_per_epoch += len(train_loader)
                # batch_per_epoch = len(self.train_loader)
                t_total = int(batch_per_epoch * args.epochs)
                print("BertAdam Total Iters: %d" % t_total)
                from lxrt.optimization import BertAdam
                self.optim = BertAdam(list(self.model.parameters()),
                                      lr=args.lr,
                                      warmup=0.1,
                                      t_total=t_total)
            else:
                self.optim = args.optimizer(self.model.parameters(), args.lr)
            # Output Directory
            now = datetime.datetime.now()
            month_day_hour = f"{now.month:02d}{now.day:02d}{now.hour:02d}"
            f = "forward"
            if args.cl_reverse:
                f = "reversed"

            self.output = os.path.join(args.output,
                                       f"{args.dataset}_{args.mode}_{args.cl}_{args.ratio}_{f}_{month_day_hour}")
            # self.output = args.output
            os.makedirs(self.output, exist_ok=True)

    def train(self, train_loader_lst, val_loader, qid2type, args):
        best_valid = 0.
        logger = utils.Logger(os.path.join(self.output, 'log.txt'))
        mode = args.mode

        qtype_lst = [train_loader.dataset.qtype for train_loader in train_loader_lst]
        qtype_dist_diff_dict = {el: 0 for el in qtype_lst}

        self.model, self.optim, qtype_dist_diff_dict = warmup_model(self.model, train_loader_lst, self.optim, args,
                                                          qtype_dist_diff_dict, logger)
        # train
        total_num = sum(len(train_loader.dataset) for train_loader in train_loader_lst)
        comp_list = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
        for comp in comp_list:
            current_cl, current_cl_epoch = get_cl_schedule(train_loader_lst, comp, qtype_dist_diff_dict, args)
            # current_cl, current_cl_epoch, qtype_score_dict = get_curriculum_schedule(train_loader_lst, diffculty_model, comp, args)

            cl_qtype_lst = [train_loader.dataset.qtype for train_loader in current_cl]
            logger.write("Length of the training questions :  " + str(len(cl_qtype_lst)))
            logger.write('Qtype Curriculm List:  %s' % (cl_qtype_lst))
            logger.write('Qtype Scores List:  %s' % (qtype_dist_diff_dict))

            qtype_score_lst_dict = {el: [] for el in qtype_lst}
            qtype_ids_lst_dict = {el: [] for el in qtype_lst}
            qtype_loss_lst_dict = {el: [] for el in qtype_lst}

            if comp == 1:
                current_cl_epoch = 8

            for epoch in range(current_cl_epoch):
                total_loss = 0
                total_bce_loss = 0
                total_con_loss = 0
                train_score_pos = 0
                for train_loader in current_cl:
                    for i, (feats, boxes, sent, sent_pos, target, ques_id) in tqdm(enumerate(train_loader)):
                        self.model.train()
                        self.optim.zero_grad()
                        batch_size = feats.size(0)
                        feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                        if mode == 'lxmert':
                            out_dict = self.model(feats, boxes, list(sent))
                            # base VQA model
                            bce_loss = instance_bce_with_logits(out_dict['logits'], target, reduction='mean')
                            loss = bce_loss
                        total_loss += loss.item()
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                        self.optim.step()
                        score_pos = compute_score_with_logits(out_dict['logits'], target.data).sum()
                        train_score_pos += score_pos.item()
                        total_loss += loss.item() * batch_size
                        total_bce_loss += bce_loss.item() * batch_size

                # evaluate the model on the training data after the epoch end
                self.model.train(False)
                for train_loader in train_loader_lst:
                    train_results = evaluate_train(self.model, train_loader, args)
                    qtype_score_lst_dict[train_loader.dataset.qtype].append(train_results['scores_lst'])
                    qtype_ids_lst_dict[train_loader.dataset.qtype].append(train_results['ids_lst'])
                    qtype_loss_lst_dict[train_loader.dataset.qtype].append(train_results['total_loss_lst'])
                self.model.train(True)
                self.save("LAST")


                # if self.valid_tuple is not None:  # Do Validation
                self.model.train(False)
                results = evaluate(self.model, val_loader, qid2type)

                eval_score = results["score"]
                bound = results["upper_bound"]
                entropy = results['entropy']
                yn = results['score_yesno']
                other = results['score_other']
                num = results['score_number']

                self.model.train(True)
                if eval_score > best_valid:
                    best_valid = eval_score
                    self.save("BEST")

                log_str = "Epoch %d: Valid %0.2f\n" % (epoch, eval_score * 100.) + \
                            "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

                print(log_str)
                if mode == 'lxmert_q_con':
                    total_con_loss = total_con_loss / total_num

                logger.write('Epoch %d: train_loss: %.2f, BCE_loss: %.2f,'
                         ' con_loss: %.2f, '
                         'score: %.2f'
                         % (epoch, total_loss / total_num, total_bce_loss / total_num,
                            total_con_loss,
                            (train_score_pos / total_num) * 100))

                logger.write( '\ttrain_loss: %.2f, score: %.2f' % (total_loss/total_num, (train_score_pos/total_num) * 100))
                if val_loader is not None:
                    logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
                    logger.write('\tyn score: %.2f other score: %.2f num score: %.2f' % (100 * yn, 100 * other, 100 * num))

                if val_loader is not None and entropy is not None:
                    info = '' + ' %.2f' % entropy
                    logger.write('\tentropy: ' + info)

                with open(self.output + "/log.log", 'a') as f:
                    f.write(log_str)
                    f.flush()


            print(type(qtype_score_lst_dict))
            print("qtype_score_lst_dict: len", len(qtype_score_lst_dict))
            print("qtype_score_lst_dict: ", qtype_score_lst_dict.keys())

            # calculate the stats for the model to decide the next CL
            # qtype_mean_dict, qtype_median_dict, qtype_sd_dict = do_stats(qtype_score_lst_dict)
            qtype_dist_diff_dict = compute_dist_diff(qtype_loss_lst_dict, qtype_dist_diff_dict, args)

        return best_valid

    def save(self, name):
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(model_to_save.state_dict(), os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)


if __name__ == "__main__":
    vqa = VQA()
    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        vqa.load(args.load)
    
    with open('/scratch3/akl002/VQA/dvqa39/util/qid2type_%s.json' % args.dataset, 'r') as f:
        qid2type = json.load(f)
    vqa.train(vqa.train_loader_lst, vqa.val_loader, qid2type, args)
