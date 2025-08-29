import sys
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.init as init
import numpy as np

from dataset_tpcl import Dictionary, VQAFeatureDataset

import utils
import opts
from train_tpcl_dyn import train_cl
import os
import json
import datetime
from SAN import Model

def weights_init_kn(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, a=0.01)

if __name__ == '__main__':
    opt = opts.parse_opt()
    dataset = opt.dataset
    now = datetime.datetime.now()
    month_day_hour = f"{now.month:02d}{now.day:02d}{now.hour:02d}"
    opt.output = os.path.join(opt.output, f"{opt.dataset}_{opt.mode}_{opt.cl}_{month_day_hour}")

    if not os.path.isdir(opt.output):
        utils.create_dir(opt.output)

    seed = 0
    if opt.seed == 0:
        seed = random.randint(1, 10000)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(opt.seed)
    else:
        seed = opt.seed
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file(opt.dataroot + '/dictionary.pkl')
    opt.ntokens = dictionary.ntoken

    model = Model(opt)
    model = model.cuda()
    model.apply(weights_init_kn)
    model = nn.DataParallel(model).cuda()

    with open('util/qid2type_%s.json' % opt.dataset, 'r') as f:
        qid2type = json.load(f)

    # load curriculum
    if opt.cl == "ling_diff_type":
        selected_cl = json.load(open('cl/cl_order.json'))['curriculum']['linguistic']
    else:
        selected_cl = json.load(open('cl/cl_order.json'))['curriculum'][opt.cl]

    print("selected cl: ", selected_cl)
    print("Building train dataset...")
    VQAFeats_lst = []

    for qtype in selected_cl:
        print("qtype: ", qtype)
        clss_obj = VQAFeatureDataset('train', dictionary, opt.dataroot, opt.img_root, ratio=opt.ratio, qtype=qtype,
                                    adaptive=False, dataset=opt.dataset)  # load labeld data
        print("data len: ", len(clss_obj))
        VQAFeats_lst.append(clss_obj)

    print("Building test dataset...")
    # change data name into val in case of v2
    test_dset = VQAFeatureDataset('test', dictionary, opt.dataroot, opt.img_root, ratio=1.0, adaptive=False,
                      dataset=opt.dataset)
    print("Building train dataset loader...")
    train_loader_lst = []
    for VQA_obj in VQAFeats_lst:
        train_loader_lst.append(DataLoader(VQA_obj, opt.batch_size, shuffle=True, num_workers=4, collate_fn=utils.trim_collate))

    eval_loader = DataLoader(test_dset, opt.batch_size, shuffle=False, num_workers=4, collate_fn=utils.trim_collate)

    print("Starting training...")
    train_cl(model, train_loader_lst, eval_loader, opt, qid2type)