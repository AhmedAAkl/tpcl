import copy
import os
import time
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import json
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn import CosineSimilarity
import torch.nn.functional as F
from torch.utils.collect_env import get_pretty_env_info
import random
from info_nce import InfoNCE, info_nce
import numpy as np
import statistics as stat
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import ot
import matplotlib.pyplot as plt


# standard cross-entropy loss
def instance_bce(logits, labels):
    assert logits.dim() == 2
    cross_entropy_loss = nn.CrossEntropyLoss()

    prediction_ans_k, top_ans_ind = torch.topk(
        F.softmax(labels, dim=-1), k=1, dim=-1, sorted=False)
    ce_loss = cross_entropy_loss(logits, top_ans_ind.squeeze(-1))

    return ce_loss
# multi-label soft loss
def instance_bce_with_logits(logits, labels, reduction='mean'):
    assert logits.dim() == 2
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction=reduction)
    if reduction == 'mean':
        loss *= labels.size(1)
    return loss
def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores
def compute_OT(data_dict, qtype_dist_diff_dict, opt, warmp_up=True):
    weights = [0.1, 0.1, 0.3, 0.5]
    qtype_lst = data_dict.keys()
    for qtype, scores_lst in data_dict.items(): # iterate over qtype scores dict.
        qtype_dist_diff_lst = []
        for i in range(len(scores_lst)-1):
            counts, support, _ = plt.hist(scores_lst[i], 101, (0, 100))
            counts2, support2, _ = plt.hist(scores_lst[i+1], 101, (0, 100))
            supports = (support[:-1] + support[1:]) / 2
            d = euclidean_distances(np.array([supports, supports]).T)
            prob1 = counts / np.sum(counts)
            prob2 = counts2 / np.sum(counts2)
            gamma, r = ot.emd(prob1, prob2, d, log=True)
            qtype_dist_diff_lst.append(r['cost'])
        qtype_dist_diff_dict[qtype] = sum(np.multiply(qtype_dist_diff_lst, weights))
    return qtype_dist_diff_dict

def warmup_model(model, train_loader_lst, test_loader_lst, optim, opt, qtype_dist_diff_dict, logger):
    qtype_lst = [train_loader.dataset.qtype for train_loader in train_loader_lst]
    qtype_score_lst_dict = {el: [] for el in qtype_lst}
    qtype_ids_lst_dict = {el: [] for el in qtype_lst}
    qtype_loss_lst_dict = {el: [] for el in qtype_lst}
    N = 0
    for train_loader in train_loader_lst:
        N += len(train_loader.dataset)

    logger.write("Hello from Warming up")

    for epoch in range(opt.warmup_epochs):
        train_score = 0
        total_norm = 0
        count_norm = 0
        total_loss = 0
        total_bce_loss = 0
        for train_loader in train_loader_lst:
            # features, torch.FloatTensor([0]), question, question_pos, target, question_id
            for i, (v, b, q, a, qid) in enumerate(train_loader):
                v = v.cuda()
                q = q.cuda().long()
                a = a.cuda()
                batch_size = q.size(0)
                if 'updn' in opt.mode:
                    out = model(v, q)
                    bce_loss = instance_bce_with_logits(out['logits'], a, reduction='mean')
                    loss = bce_loss
                elif 'san' in opt.mode:
                    out = model(v, q)
                    bce_loss = instance_bce_with_logits(out['logits'], a, reduction='mean')
                    loss = bce_loss
            loss.backward()
            total_norm += nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            count_norm += 1
            optim.step()
            optim.zero_grad()
            score = compute_score_with_logits(out['logits'], a.data).sum()
            train_score += score.item()
            total_loss += loss.item() * v.size(0)
            total_bce_loss += bce_loss.item() * v.size(0)
        total_loss /= N
        total_bce_loss /= N
        train_score = 100 * train_score / N
        model.train(False)
        # evaluate the model on the training data after the epoch end
        for train_loader in train_loader_lst:
            train_results = evaluate_train(model, train_loader, opt)
            qtype_score_lst_dict[train_loader.dataset.qtype].append(train_results['scores_lst'])
            qtype_ids_lst_dict[train_loader.dataset.qtype].append(train_results['ids_lst'])
            qtype_loss_lst_dict[train_loader.dataset.qtype].append(train_results['total_loss_lst'])
        model.train(True)
        logger.write("Warming up Epoch:  " + str(epoch))

        # activate to shuffle the question type order
        # if opt.cl_shuffle:
        #     random.shuffle(train_loader_lst)

    qtype_dist_diff_dict = compute_OT(qtype_loss_lst_dict, qtype_dist_diff_dict, opt, warmp_up=True)

    return model, optim, qtype_dist_diff_dict
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


def train_cl(model, train_loader_lst, eval_loader, opt, qid2type):
    # utils.create_dir(opt.output)
    optim = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999), eps=1e-08,
                             weight_decay=opt.weight_decay)
    logger = utils.Logger(os.path.join(opt.output, 'log.txt'))
    utils.print_model(model, logger)
    # load snapshot
    if opt.checkpoint_path is not None:
        print('loading %s' % opt.checkpoint_path)
        model_data = torch.load(opt.checkpoint_path)
        model.load_state_dict(model_data.get('model_state', model_data))
        optim.load_state_dict(model_data.get('optimizer_state', model_data))
        opt.s_epoch = model_data['epoch'] + 1

    for param_group in optim.param_groups:
        param_group['lr'] = opt.learning_rate
    scheduler = MultiStepLR(optim, milestones=[10, 15, 20, 25], gamma=0.5)
    scheduler.last_epoch = opt.s_epoch
    qtype_lst = [train_loader.dataset.qtype for train_loader in train_loader_lst]
    qtype_dist_diff_dict = {el: 0 for el in qtype_lst}
    model, optim, qtype_dist_diff_dict = warmup_model(model, train_loader_lst, eval_loader, optim, opt, qtype_dist_diff_dict, logger)
    best_eval_score = 0
    N = sum(len(train_loader.dataset) for train_loader in train_loader_lst)
    pacing_val = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
    # print("Pacing Value: ", pacing_val)
    for comp in pacing_val:
        current_cl, current_cl_epoch = get_cl_schedule(train_loader_lst, comp, qtype_dist_diff_dict, opt)

        cl_qtype_lst = [train_loader.dataset.qtype for train_loader in current_cl]
        logger.write("Length of the training questions :  " + str(len(cl_qtype_lst)))
        logger.write("Competence Values :  " + str(comp))
        logger.write('Qtype List:  %s' % (cl_qtype_lst))
        logger.write('Qtype Scores dict:  %s' % (qtype_dist_diff_dict))

        qtype_score_lst_dict = {el: [] for el in qtype_lst}
        qtype_ids_lst_dict = {el: [] for el in qtype_lst}
        qtype_loss_lst_dict = {el: [] for el in qtype_lst}

        for epoch in np.arange(current_cl_epoch):
            total_loss = 0
            total_bce_loss = 0
            train_score_pos = 0
            total_norm = 0
            count_norm = 0
            t = time.time()
            if epoch == 4:
                scheduler.step()
            mode = opt.mode
            for train_loader in current_cl:
                for i, (v, b, q, q_pos, a, _) in enumerate(train_loader):
                    v = v.cuda()
                    q = q.cuda().long()
                    q_pos = q_pos.cuda().long()
                    a = a.cuda()
                    batch_size = q.size(0)
                    if mode == 'updn':
                        out = model(v, q)
                        bce_loss = instance_bce_with_logits(out['logits'], a, reduction='mean')
                        loss = bce_loss
                    elif mode == 'san':
                        out = model(v, q)
                        bce_loss = instance_bce_with_logits(out['logits'], a, reduction='mean')
                        loss = bce_loss
                    loss.backward()
                    total_norm += nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
                    count_norm += 1
                    optim.step()
                    optim.zero_grad()
                    score_pos = compute_score_with_logits(out['logits'], a.data).sum()
                    train_score_pos += score_pos.item()
                    total_loss += loss.item() * v.size(0)
                    total_bce_loss += bce_loss.item() * v.size(0)
            # evaluate the model on the training data after the epoch end
            model.train(False)
            for train_loader in train_loader_lst:
                train_results = evaluate_train(model, train_loader, opt)
                qtype_score_lst_dict[train_loader.dataset.qtype].append(train_results['scores_lst'])
                qtype_ids_lst_dict[train_loader.dataset.qtype].append(train_results['ids_lst'])
                qtype_loss_lst_dict[train_loader.dataset.qtype].append(train_results['total_loss_lst'])
            model.train(True)

            total_loss /= N
            total_bce_loss /= N
            train_score_pos = 100 * train_score_pos / N
            if None != eval_loader:
                model.train(False)
                results = evaluate(model, eval_loader, qid2type, opt)
                model.train(True)
                eval_score = results["score"]
                bound = results["upper_bound"]
                entropy = results['entropy']
                yn = results['score_yesno']
                other = results['score_other']
                num = results['score_number']
            logger.write('\nlr: %.7f' % optim.param_groups[0]['lr'])
            logger.write('epoch %d, time: %.2f' % (epoch, time.time() - t))
            logger.write('\ttrain_loss: %.2f, BCE_loss: %.2f, score: %.2f '
                         % (total_loss, total_bce_loss,
                            train_score_pos))
            if eval_loader is not None:
                logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
                logger.write('\tyn score: %.2f other score: %.2f num score: %.2f' % (100 * yn, 100 * other, 100 * num))

            if eval_loader is not None and entropy is not None:
                info = '' + ' %.2f' % entropy
                logger.write('\tentropy: ' + info)

            if (eval_loader is not None and eval_score > best_eval_score):
                model_path = os.path.join(opt.output, 'best_model.pth')
                utils.save_model(model_path, model, epoch, optim)
                if eval_loader is not None:
                    best_eval_score = eval_score
        # calculate the OT for the model to decide the next CL
        if comp != 1.0:
            qtype_dist_diff_dict = compute_OT(qtype_loss_lst_dict, qtype_dist_diff_dict, opt)

@torch.no_grad()
def evaluate(model, dataloader, qid2type, opt):
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
    mode = opt.mode
    for i, (v, b, q, a, q_id) in enumerate(dataloader):
        v = v.cuda()
        b = b.cuda()
        q = q.cuda().long()
        q_id = q_id.cuda()
        if 'updn' in mode:
            out = model(v, q)
        else:
            out = model(v, q)
        batch_score = compute_score_with_logits(out['logits'], a.cuda()).cpu().numpy().sum(1)
        score += batch_score.sum()
        upper_bound += (a.max(1)[0]).sum().item()
        num_data += out['logits'].size(0)
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
    if entropy is not None:
        entropy = entropy / len(dataloader.dataset)
    results = dict(
        score=score,
        upper_bound=upper_bound,
        entropy=entropy,
        score_yesno=score_yesno,
        score_other=score_other,
        score_number=score_number,
    )
    return results

@torch.no_grad()
def evaluate_train(model, dataloader, opt):
    score = 0
    upper_bound = 0
    mode = opt.mode
    scores_lst = []
    total_loss_lst = []
    ids_lst = []
    for i, (v, b, q, a, q_id) in enumerate(dataloader):
        v = v.cuda()
        b = b.cuda()
        q = q.cuda().long()
        q_id = q_id.cuda()
        a = a.cuda()
        if 'updn' in mode:
            out = model(v, q)
        else:
            out = model(v, q)
        bce_loss = instance_bce_with_logits(out['logits'], a, reduction='none')
        batch_score = compute_score_with_logits(out['logits'], a.cuda()).cpu().numpy().sum(1)
        scores_lst.extend(batch_score.tolist())  # add scores to list
        total_loss_lst.extend(bce_loss.sum(dim=1).tolist())
        ids_lst.extend(q_id.tolist())
        score += batch_score.sum()
        upper_bound += (a.max(1)[0]).sum().item()
    if len(dataloader.dataset) != 0:
        score = score / len(dataloader.dataset)
        upper_bound = upper_bound / len(dataloader.dataset)
    else:
        score = "NAN"
    results = dict(
        score=score,
        upper_bound=upper_bound,
        scores_lst=scores_lst,
        ids_lst=ids_lst,
        total_loss_lst=total_loss_lst,
    )
    return results
