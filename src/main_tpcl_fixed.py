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
process = psutil.Process(os.getpid())

import torch.nn.functional as F
import json
import pickle as cPickle  # python3
import utils
import datetime
import random

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
        out_dict = model(v, b, q, None, None, False)
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

def get_qtypes(dataroot, name="train", img_id2val=None, dataset="cpv2"):
    """Load entries
    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    if dataset == 'cpv2':
        answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
        name = "train" if name == "train" else "test"
    elif dataset == 'cpv1':
        answer_path = os.path.join(dataroot, 'cp-v1-cache', '%s_target.pkl' % name)
        name = "train" if name == "train" else "test"
        question_path = os.path.join(dataroot, 'vqacp_v1_%s_questions.json' % name)
        with open(question_path) as f:
            questions = json.load(f)
    elif dataset == 'v2':
        answer_path = os.path.join(dataroot, 'cache_v2', '%s_target.pkl' % name)

    with open(answer_path, 'rb') as f:
        answers = cPickle.load(f)

    answers.sort(key=lambda x: x['question_id'])

    questions_types = []
    for ans in answers:
        questions_types.append(ans['question_type'])

    entries_dict = {el: 0 for el in questions_types}

    return entries_dict

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
    # for qtype in list(train_qtype_dict.keys()):
    for qtype in selected_cl:
        print("qtype: ", qtype)
        VQAFeats_lst.append(VQAFeatureDataset('train', dictionary, opt.dataroot, opt.img_root, ratio=opt.ratio,
                                              qtype=qtype,adaptive=False, dataset=opt.dataset))


    train_qtype_lst = []
    for VQA_obj in VQAFeats_lst:
        train_qtype_lst.append(DataLoader(VQA_obj, opt.batch_size, shuffle=True, num_workers=4, collate_fn=utils_1.trim_collate))

    test_qtypes = get_qtypes(opt.dataroot, name='test')
    print("Building Test dataset loader...")
    feats_test_lst = []
    for qtype in test_qtypes.keys():
        feats_test_lst.append(VQAFeatureDataset('test', dictionary, opt.dataroot, opt.img_root,ratio=1.0, adaptive=False,
                                      qtype= qtype, dataset=opt.dataset))

    print("Building Test dataset loader...")
    eval_dset = VQAFeatureDataset('test', dictionary, opt.dataroot, opt.img_root, ratio=1.0, adaptive=False,
                                  dataset=opt.dataset)
    opt.use_all = 1
    val_loader = DataLoader(eval_dset, opt.batch_size, shuffle=False, num_workers=4, collate_fn=utils_1.trim_collate)

    return train_qtype_lst, val_loader

class VQA:
    def __init__(self,folder="/",load=True):
        # Datasets
        self.train_loader_lst, self.val_loader_lst = get_our_data()
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
            self.output = os.path.join(args.output, f"{args.dataset}_{args.mode}_{args.cl}_{args.ratio}_{month_day_hour}")
            # self.output = args.output
            os.makedirs(self.output, exist_ok=True)

    def train(self, train_loader_lst, val_loader, qid2type, args):
        best_valid = 0.
        logger = utils.Logger(os.path.join(self.output, 'log.txt'))
        mode = args.mode
        # train
        total_num = 0
        for train_loader in train_loader_lst:
            total_num += len(train_loader.dataset)

        if args.cl == 'ling_diff_type':
            # load curriculum
            selected_cl = json.load(open('../cl/cl_order.json'))['curriculum'][args.cl]
            comp_list = selected_cl
            qtype_ling = list(comp_list.keys())

            for i in range(1, len(qtype_ling) + 1):
                current_cl = []
                current_qtype_lst = qtype_ling[:i]
                if args.cl == 'ling_diff_type':
                    for train_loader in train_loader_lst:
                        for key in current_qtype_lst:
                            if train_loader.dataset.qtype in comp_list[key]:
                                current_cl.append(train_loader)

                current_cl_epoch = 5

                logger.write("Length of the training questions :  " + str(len(current_cl)))

                for epoch in range(current_cl_epoch):
                    total_loss = 0
                    total_bce_loss = 0
                    total_con_loss = 0
                    train_score_pos = 0
                    for train_loader in current_cl:
                        for i, (feats, boxes, sent, target, ques_id) in tqdm(enumerate(train_loader)):

                            self.model.train()
                            self.optim.zero_grad()
                            batch_size = feats.size(0)

                            feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()

                            if mode == 'lxmert':
                                # print("list of sent: ", list(sent))
                                out_dict = self.model(feats, boxes, list(sent), None, None, self_sup=False)
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

                    self.save("LAST")
                    qtype_lst = []
                    for train_loader in train_loader_lst:
                        # N += len(train_loader.dataset)
                        qtype_lst.append(train_loader.dataset.qtype)

                    logger.write('Qtype List:  %s' % (qtype_lst))

                    # if self.valid_tuple is not None:  # Do Validation
                    self.model.train(False)
                    results = evaluate(self.model, val_loader, qid2type)

                    eval_score = results["score"]
                    bound = results["upper_bound"]
                    entropy = results['entropy']
                    yn = results['score_yesno']
                    other = results['score_other']
                    num = results['score_number']

                    # eval_score = results['score']
                    # valid_score, upper_bound = evaluate(self.model, val_loader)
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

                    # if epoch == 9:
                    #     break

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
    vqa.train(vqa.train_loader_lst, vqa.val_loader_lst, qid2type, args)
