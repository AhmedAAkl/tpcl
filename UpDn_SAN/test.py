"""
This code is copied from SSL-VQA's repository.
https://github.com/CrossmodalGroup/SSL-VQA
"""
import argparse
import json
import progressbar
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset_tpcl import Dictionary, VQAFeatureDataset
from UpDn import BaseModel
import utils
import opts
import tqdm


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def get_question(q, dataloader):
    str = []
    dictionary = dataloader.dataset.dictionary
    for i in range(q.size(0)):
        str.append(dictionary.idx2word[q[i]] if q[i] < len(dictionary.idx2word) else '_')
    return ' '.join(str)


def get_answer(p, dataloader):
    _m, idx = p.max(0)
    return dataloader.dataset.label2ans[idx.item()]


@torch.no_grad()
def get_logits(model, dataloader):
    N = len(dataloader.dataset)
    M = dataloader.dataset.num_ans_candidates
    K = 36
    pred = torch.FloatTensor(N, M).zero_()
    qIds = torch.IntTensor(N).zero_()
    idx = 0
    bar = progressbar.ProgressBar(maxval=N or None).start()
    for v, b, q, a, i in iter(dataloader):
        bar.update(idx)
        batch_size = v.size(0)
        v = v.cuda()
        b = b.cuda()
        q = q.cuda()
        out = model(v,q,False)
        pred[idx:idx+batch_size,:].copy_(out['logits'].data)
        qIds[idx:idx+batch_size].copy_(i)
        idx += batch_size

    bar.update(idx)
    return pred, qIds


def make_json(logits, qIds, dataloader):
    utils.assert_eq(logits.size(0), len(qIds))
 
    results = []
    for i in range(logits.size(0)):
        result = {}
        result['question_id'] = qIds[i].item()
        result['answer'] = get_answer(logits[i], dataloader)
        results.append(result)
    return results


def evaluate(model,dataloader,qid2type, opt):
    score = 0
    upper_bound = 0
    score_yesno = 0
    score_number = 0
    score_other = 0
    total_yesno = 0
    total_number = 0
    total_other = 0
    num_data = 0
    
    model.train(False)
    # import pdb;pdb.set_trace()
    mode = opt.mode


    for i, (v, b, q, a, q_id) in enumerate(dataloader):
        v = v.cuda()
        q = q.cuda()
        out_pos = model(v, q)
        batch_score = compute_score_with_logits(out_pos['logits'], a.cuda()).cpu().numpy().sum(1)
        # score += batch_score.item()
        score += batch_score.sum()
        upper_bound += (a.max(1)[0]).sum().item()
        num_data += out_pos['logits'].size(0)
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
    print('\teval overall score: %.2f' % (100 * score))
    print('\teval up_bound score: %.2f' % (100 * upper_bound))
    print('\teval y/n score: %.2f' % (100 * score_yesno))
    print('\teval other score: %.2f' % (100 * score_other))
    print('\teval number score: %.2f' % (100 * score_number))



if __name__ == '__main__':
    opt = opts.parse_opt()

    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file(opt.dataroot + 'dictionary.pkl')
    opt.ntokens = dictionary.ntoken
    eval_dset = VQAFeatureDataset('test', dictionary, opt.dataroot, opt.img_root, 1.0, adaptive=False)

    n_device = torch.cuda.device_count()
    batch_size = opt.batch_size * n_device

    model = BaseModel(opt)
    model = model.cuda()

    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=1, collate_fn=utils.trim_collate)
    def process(args, model, eval_loader):

        print('loading %s' % opt.checkpoint_path)
        model_data = torch.load(opt.checkpoint_path)

        model = nn.DataParallel(model).cuda()
        model.load_state_dict(model_data.get('model_state', model_data))
        opt.s_epoch = model_data['epoch'] + 1

        model.train(False)

        with open('util/qid2type_%s.json' % opt.dataset, 'r') as f:
            qid2type = json.load(f)

        evaluate(model,eval_loader,qid2type, opt)

    process(opt, model, eval_loader)
