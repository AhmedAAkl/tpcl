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
# from info_nce import InfoNCE, info_nce
import numpy as np
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

def compute_self_loss(logits_neg, a):
    prediction_ans_k, top_ans_ind = torch.topk(
        F.softmax(a, dim=-1), k=1, dim=-1, sorted=False)
    neg_top_k = torch.gather(
        F.softmax(logits_neg, dim=-1), 1, top_ans_ind).sum(1)

    qice_loss = neg_top_k.mean()
    return qice_loss

def get_curriculum_schedule(train_qtype_lst, model, comp, opt):
    dict_q = {}

    if model is not None:
        model.eval()

        for loader in train_qtype_lst:
            train_results = evaluate_train(model, loader, None, opt)
            dict_q[loader.dataset.qtype] = 1- train_results['score']
        qtype_score_dict = dict(filter(lambda item: item[1] >= comp, dict_q.items()))
    else:
        qtype_score_dict = json.load(open('qtype_scores_dicts.json'))
        qtype_score_dict = dict(filter(lambda item: 1 - item[1]['score'] >= comp, qtype_score_dict.items()))

    cl_list = []
    for k, v in qtype_score_dict.items():
        for item in train_qtype_lst:
            if item.dataset.qtype == k:
                cl_list.append(item)

    if model is not None:
        model.train(True)

    if comp == 0:
        current_cl_epoch = 10
    else:
        current_cl_epoch = 5

    return cl_list, current_cl_epoch


def train_cl(model, train_qtype_lst, eval_loader, opt, qid2type):
    # utils.create_dir(opt.output)
    if opt.cl_reordering:
        qtype_lst = [train_loader.dataset.qtype for train_loader in train_qtype_lst]
        qtype_score_dict = {el: 0 for el in qtype_lst}
    else:
        qtype_score_dict = json.load(open('qtype_scores_dicts.json'))

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

    best_eval_score = 0
    N = 0
    for train_loader in train_qtype_lst:
        N += len(train_loader.dataset)

    total_loss_lst = []
    total_bce_lst = []
    total_con_lst = []
    comp_list = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
    if opt.cl  == 'ling_diff_type':
        # load curriculum
        selected_cl = json.load(open('cl/cl_order.json'))['curriculum'][opt.cl]
        comp_list = selected_cl
        qtype_ling = list(selected_cl.keys())

    else:
        selected_cl = json.load(open('cl/cl_order.json'))['curriculum'][opt.cl]

    for i in range(1, len(qtype_ling) + 1):
        current_cl = []
        current_qtype_lst = qtype_ling[:i]
        if opt.cl == 'ling_diff_type':
            for train_loader in train_qtype_lst:
                for key in current_qtype_lst:
                    if train_loader.dataset.qtype in comp_list[key]:
                        current_cl.append(train_loader)
        for train_loader in train_qtype_lst:
            if train_loader.dataset.qtype in current_qtype_lst: #comp_list[key]:
                current_cl.append(train_loader)
    # else:
    #     if comp == comp_list[0]:
    #         if opt.cl_diff_model_path is not None:
    #             print('loading CL_DIFF Pretarined model %s' % opt.cl_diff_model_path)
    #             # diffculty_model = copy.deepcopy(model)
    #             # model_data = torch.load(opt.cl_diff_model_path)
    #             # diffculty_model.load_state_dict(model_data.get('model_state', model_data))
    #             # diffculty_model.load_state_dict(model_data.get('optimizer_state', model_data))
    #
    #             model_data = torch.load(opt.cl_diff_model_path)
    #             model.load_state_dict(model_data.get('model_state', model_data))
    #             optim.load_state_dict(model_data.get('optimizer_state', model_data))
    #             diffculty_model = model
    #         else:
    #             # print("Pretrained diff model needed for CL mode !!")
    #             print("Use the scores from the pretrained model")
    #             diffculty_model = None
    #             # exit()exit
    #     else:
    #         diffculty_model = model
    #
    #     current_cl, current_cl_epoch = get_curriculum_schedule(train_qtype_lst, diffculty_model, comp, opt)

        current_cl_epoch = 10
        qtype_lst = [train_loader.dataset.qtype for train_loader in current_cl]
        logger.write("Length of the training questions :  " +  str(len(qtype_lst)))
        logger.write('Qtype List:  %s' % (qtype_lst))
        for epoch in np.arange(current_cl_epoch):
            total_loss = 0
            total_bce_loss = 0
            train_score_pos = 0
            total_con_loss = 0
            total_norm = 0
            count_norm = 0
            t = time.time()
            scheduler.step()
            mode = opt.mode
            for train_loader in current_cl:
                for i, (v, b, q, a, q_id) in enumerate(train_loader):
                    v = v.cuda()
                    q = q.cuda().long()
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
                # evaluate the score for the training data
                model.eval()
                current_cl_results = evaluate_train(model, train_loader, qid2type, opt)
                qtype_score_dict[train_loader.dataset.qtype] = current_cl_results['score']
                model.train(True)

            total_loss /= N
            total_bce_loss /= N
            total_con_loss /= N
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
                            train_score_pos ))

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

            if opt.cl_reordering:
                # sort the data list based on the training score
                # normal from low to high, use reverse from high to low
                qtype_score_dict = dict(sorted(qtype_score_dict.items(), key=lambda item: item[1], reverse=True))
                logger.write('Qtype Score Sorted:  %s' % (qtype_score_dict))
                temp_lst = []
                for k, v in qtype_score_dict.items():
                    for item in train_qtype_lst:
                        if item.dataset.qtype == k:
                            temp_lst.append(item)

                train_qtype_lst = temp_lst
                del temp_lst


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
def evaluate_train(model, dataloader, qid2type, opt):
    score = 0
    upper_bound = 0

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
    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)

    results = dict(
        score=score,
        upper_bound=upper_bound,
    )
    return results
