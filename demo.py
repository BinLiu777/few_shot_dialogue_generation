from __future__ import print_function

import warnings
warnings.filterwarnings('ignore')

import argparse
import os
import torch
import sys
import random
import logging
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'NeuralDialog_ZSDG'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'NeuralDialog_LAED'))

# TODO:解决引入冲突
from NeuralDialog_LAED.laed.dataset import data_loaders
from NeuralDialog_LAED.laed.utils import str2bool, prepare_dirs_loggers, process_config
from NeuralDialog_LAED.laed.enc2dec.decoders import GEN
from models.models import LAPtrHRED, ZeroShotLAPtrHRED
from utils import corpora
from utils.corpora import load_vocab, load_model, load_config

from NeuralDialog_ZSDG.zsdg import evaluators
from utils import data_loaders as fsdg_data_loaders
from NeuralDialog_ZSDG.zsdg import hred_utils

arg_lists = []
parser = argparse.ArgumentParser()
logger = logging.getLogger()


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


def process_data_feed(model, feed, config):
    features = []
    model.eval()
    feed.epoch_init(config, shuffle=False, verbose=True)
    while True:
        batch = feed.next_batch()
        if batch is None:
            break
        laed_z = model.forward(batch, GEN, gen_type=config.gen_type, return_latent=True)
        features.append(laed_z.data.cpu().numpy())
    return np.array(features).reshape(-1, config.y_size)
    # return np.array(features).reshape(-1, config.y_size * config.k)


def deflatten_laed_features(in_laed_features, in_dialogs, pad_mode=None):
    pad = np.zeros_like(in_laed_features[0])
    result = []
    start_turn = 0
    for dialog_i in in_dialogs:
        if pad_mode == 'start_end':
            dialog_i_turns = np.concatenate([[pad],
                                             in_laed_features[start_turn: start_turn + len(dialog_i) - 2,:],
                                             [pad]],
                                            axis=0)
            start_turn += len(dialog_i) - 2
        elif pad_mode == 'start':
            dialog_i_turns = np.concatenate([[pad],
                                             in_laed_features[start_turn: start_turn + len(dialog_i) - 1,:]],
                                            axis=0)
            start_turn += len(dialog_i) - 1
        else:
            dialog_i_turns = in_laed_features[start_turn: start_turn + len(dialog_i),:]
            start_turn += len(dialog_i)
        while len(dialog_i) < dialog_i_turns.shape[0]:
            dialog_i_turns = np.delete(dialog_i_turns, (-1), axis=0)
        result.append(dialog_i_turns)
    assert len(in_dialogs) == len(result)
    return result


def load_fsdg_model(config=None, corpus_client=None):
    fsdg_model = LAPtrHRED(corpus_client, config)
    if config.use_gpu:
        fsdg_model.load_state_dict(torch.load(os.path.join(config.load_sess, 'model')))
    else:
        fsdg_model.load_state_dict(torch.load(os.path.join(config.load_sess, 'model'), map_location='cpu'))
    return fsdg_model


def main(config):
    aeed_config = load_config(config.model)
    fsdg_config = load_config(config.load_sess)

    setattr(aeed_config, 'data_loader', config.data_loader)
    setattr(aeed_config, 'corpus_client', config.corpus_client)
    setattr(aeed_config, 'batch_size', config.batch_size)
    setattr(aeed_config, 'use_gpu', config.use_gpu)
    setattr(aeed_config, 'include_domain', True)
    setattr(aeed_config, 'include_example', False)
    setattr(aeed_config, 'include_state', True)
    setattr(aeed_config, 'entities_file', 'NeuralDialog_ZSDG/data/stanford/kvret_entities.json')
    setattr(aeed_config, 'action_match', True)
    setattr(aeed_config, 'data_dir', config.data_dir)
    setattr(aeed_config, 'include_eod', False) # for StED model
    setattr(aeed_config, 'lowercase', True)
    setattr(aeed_config, 'vocab', 'vocabs/maluuba.json')

    setattr(fsdg_config, 'data_dir', config.data_dir)
    setattr(fsdg_config, 'fsdg_domain_description', config.fsdg_domain_description)
    setattr(fsdg_config, 'laed_z_size', 10)
    setattr(fsdg_config, 'load_sess', config.load_sess)
    setattr(fsdg_config, 'use_gpu', config.use_gpu)
    setattr(fsdg_config, 'vocab', 'vocabs/stanford.json')
    setattr(fsdg_config, 'batch_size', 1)

    aeed_corpus_client = getattr(corpora, aeed_config.corpus_client)(aeed_config)
    aeed_corpus_client.vocab, aeed_corpus_client.rev_vocab, aeed_corpus_client.unk_id = load_vocab(aeed_config.vocab)
    prepare_dirs_loggers(config, os.path.basename(__file__))

    fsdg_corpus_client = corpora.LAZslStanfordCorpusPre(fsdg_config)
    fsdg_corpus_client.vocab, fsdg_corpus_client.rev_vocab, fsdg_corpus_client.unk_id = load_vocab(fsdg_config.vocab)

    evaluator = evaluators.BleuEntEvaluator("SMD", fsdg_corpus_client.ent_metas)


    # TODO: load sted model
    aeed_model = load_model(config.model, config.model_name, config.model_type, aeed_config, corpus_client=aeed_corpus_client)
    # TODO: load fsdg model

    fsdg_model = load_fsdg_model(fsdg_config, fsdg_corpus_client)

    utts = []
    features = [np.array([0]*10)]
    intent = ''
    while True:
        if not intent:
            while True:
                intent = input('choose intent: 1: weather, 2: schedule, 3: navigate\n').strip()
                if intent == '1':
                    intent = 'weather'
                    break
                elif intent == '2':
                    intent = 'schedule'
                    break
                elif intent == '3':
                    intent = 'navigate'
                    break
                else:
                    print('Wrong choice')

        usr_utt = input('input:')
        utts.append(usr_utt)
        # print('utts: ', utts)
        # TODO: process data
        totol_data = []
        dialogues = {}

        # dialogue: [turn, turn, turn, ...]
        dialogue = []
        for i in range(len(utts)):
            utt = utts[i]
            turn = {}
            author = 'driver' if i % 2 == 0 else 'assistant'
            # TODO: 未添加槽位信息
            data = {}
            end_dialogue = False if i != len(utts)-1 else True
            utterance = utt
            data['end_dialogue'] = end_dialogue
            data['utterance'] = utterance
            turn['turn'] = author
            turn['data'] = data
            dialogue.append(turn)

        # scenario: {kb:{items, column_names, kb_title}, task:{intent}, uuid}
        scenario = {}
        kb = {}
        items = []
        column_names = []
        kb_title = 'unknown'
        kb['items'] = items
        kb['column_names'] = column_names
        kb['kb_title'] = kb_title
        task = {}
        task['intent'] = intent
        uuid = 'unknown'

        scenario['kb'] = kb
        scenario['task'] = task
        scenario['uuid'] = uuid


        dialogues['dialogue'] = dialogue
        dialogues['scenario'] = scenario

        totol_data.append(dialogues)

        aeed_corpus_client.add_test_data(totol_data)
        dataset = aeed_corpus_client.get_corpus()['test']

        feed_data = dataset
        aeed_feed = getattr(data_loaders, aeed_config.data_loader)("Test", feed_data, aeed_config)

        # TODO: get features from aeed model
        feature = process_data_feed(aeed_model, aeed_feed, aeed_config)
        feature = np.array(feature[0])
        features.append(feature)
        # print(features)
        # print(utts)
        assert len(features) == len(utts)+1

        # TODO: get response from fsdg model
        # 将features加载到数据中

        fsdg_corpus_client.load_laed_z([[features]])

        fsdg_corpus_client.add_test_data(totol_data, features)

        test_dial = fsdg_corpus_client.get_corpus()['test']

        fsdg_feed = fsdg_data_loaders.ZslLASMDDialDataLoaderPre("Test", test_dial, fsdg_config)

        response = hred_utils.predict(fsdg_model, fsdg_feed, fsdg_config, evaluator, num_batch=None)

        # response = 'On Friday there will be clear skies in Alhambra'
        # print('bot:', ' '.join(response.split()[1:]))
        print('bot:', ' '.join(response.split()[2:]))
        utts.append(' '.join(response.split()[2:]))
        features.append(np.array([0]*10))


if __name__ == "__main__":
    # Data
    data_arg = add_argument_group('Data')
    data_arg.add_argument('model')
    data_arg.add_argument('load_sess')
    data_arg.add_argument('--model_name', required=True)
    data_arg.add_argument('--model_type', default='dialog')
    data_arg.add_argument('--data_dir', nargs='+')
    data_arg.add_argument('--corpus_client', default='ZslStanfordCorpusPre')
    data_arg.add_argument('--data_loader', required=True, help='PTBDataLoader/SMDDataLoader/SMDDialogSkipLoader')
    data_arg.add_argument('--batch_size', default=1)
    data_arg.add_argument('--vocab', default=None)
    data_arg.add_argument('--domain_description', default='annotation')
    data_arg.add_argument('--fsdg_domain_description', default='nlu')

    # MISC
    misc_arg = add_argument_group('Misc')
    misc_arg.add_argument('--use_gpu', type=str2bool, default=True)
    misc_arg.add_argument('--forward_only', type=str2bool, default=True)
    misc_arg.add_argument('--gen_type', type=str, default='greedy')
    misc_arg.add_argument('--random_seed', type=int, default=2021)

    config, unparsed = get_config()
    fix_random_seed(config.random_seed)
    main(config)
