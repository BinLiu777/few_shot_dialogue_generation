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

navigate = {
    "items": [
        {
            "poi": "Stanford Express Care",
            "distance": "3 miles",
            "traffic_info": "no traffic",
            "poi_type": "hospital",
            "address": "214 El Camino Real"
        },
        {
            "poi": "The Clement Hotel",
            "distance": "5 miles",
            "traffic_info": "no traffic",
            "poi_type": "rest stop",
            "address": "657 Ames Ave"
        },
        {
            "poi": "Comfort Inn",
            "distance": "5 miles",
            "traffic_info": "no traffic",
            "poi_type": "rest stop",
            "address": "753 University Ave"
        },
        {
            "poi": "toms house",
            "distance": "1 miles",
            "traffic_info": "no traffic",
            "poi_type": "friends house",
            "address": "580 Van Ness Ave"
        },
        {
            "poi": "Stanford Oval Parking",
            "distance": "5 miles",
            "traffic_info": "no traffic",
            "poi_type": "parking garage",
            "address": "610 Amarillo Ave"
        },
        {
            "poi": "jacks house",
            "distance": "6 miles",
            "traffic_info": "no traffic",
            "poi_type": "friends house",
            "address": "864 Almanor Ln"
        },
        {
            "poi": "5672 barringer street",
            "distance": "5 miles",
            "traffic_info": "heavy traffic",
            "poi_type": "certain address",
            "address": "5672 barringer street"
        }
    ],
    "column_names": [
        "poi",
        "poi_type",
        "address",
        "distance",
        "traffic_info"
    ],
    "kb_title": "location information"
}
weather = {
    "items": [
      {
        "monday": "windy, low of 60F, high of 80F",
        "tuesday": "dry, low of 60F, high of 70F",
        "friday": "overcast, low of 50F, high of 60F",
        "wednesday": "dew, low of 50F, high of 70F",
        "thursday": "cloudy, low of 50F, high of 60F",
        "sunday": "dew, low of 30F, high of 50F",
        "location": "downtown chicago",
        "saturday": "foggy, low of 20F, high of 30F",
        "today": "monday"
      },
      {
        "monday": "rain, low of 20F, high of 40F",
        "tuesday": "overcast, low of 30F, high of 50F",
        "friday": "hot, low of 70F, high of 90F",
        "wednesday": "overcast, low of 80F, high of 100F",
        "thursday": "foggy, low of 50F, high of 70F",
        "sunday": "hail, low of 30F, high of 40F",
        "location": "carson",
        "saturday": "raining, low of 30F, high of 40F",
        "today": "monday"
      },
      {
        "monday": "drizzle, low of 30F, high of 50F",
        "tuesday": "clear skies, low of 40F, high of 50F",
        "friday": "raining, low of 20F, high of 40F",
        "wednesday": "cloudy, low of 40F, high of 50F",
        "thursday": "clear skies, low of 50F, high of 60F",
        "sunday": "cloudy, low of 90F, high of 100F",
        "location": "seattle",
        "saturday": "foggy, low of 70F, high of 80F",
        "today": "monday"
      },
      {
        "monday": "blizzard, low of 20F, high of 40F",
        "tuesday": "dew, low of 60F, high of 70F",
        "friday": "drizzle, low of 70F, high of 90F",
        "wednesday": "clear skies, low of 90F, high of 100F",
        "thursday": "dew, low of 90F, high of 100F",
        "sunday": "rain, low of 60F, high of 70F",
        "location": "alhambra",
        "saturday": "hot, low of 80F, high of 100F",
        "today": "monday"
      },
      {
        "monday": "stormy, low of 80F, high of 100F",
        "tuesday": "blizzard, low of 30F, high of 50F",
        "friday": "overcast, low of 30F, high of 50F",
        "wednesday": "raining, low of 70F, high of 90F",
        "thursday": "clear skies, low of 70F, high of 90F",
        "sunday": "overcast, low of 60F, high of 80F",
        "location": "mountain view",
        "saturday": "stormy, low of 20F, high of 40F",
        "today": "monday"
      },
      {
        "monday": "cloudy, low of 60F, high of 70F",
        "tuesday": "overcast, low of 40F, high of 50F",
        "friday": "raining, low of 90F, high of 100F",
        "wednesday": "clear skies, low of 80F, high of 90F",
        "thursday": "windy, low of 80F, high of 100F",
        "sunday": "raining, low of 60F, high of 80F",
        "location": "grand rapids",
        "saturday": "foggy, low of 80F, high of 90F",
        "today": "monday"
      },
      {
        "monday": "rain, low of 40F, high of 50F",
        "tuesday": "clear skies, low of 30F, high of 40F",
        "friday": "clear skies, low of 40F, high of 60F",
        "wednesday": "hail, low of 20F, high of 30F",
        "thursday": "clear skies, low of 20F, high of 30F",
        "sunday": "rain, low of 20F, high of 30F",
        "location": "redwood city",
        "saturday": "frost, low of 40F, high of 60F",
        "today": "monday"
      }
    ],
    "column_names": [
      "location",
      "monday",
      "tuesday",
      "wednesday",
      "thursday",
      "friday",
      "saturday",
      "sunday",
      "today"
    ],
    "kb_title": "weekly forecast"
  }
schedule = {
    "items": [
      {
        "room": "-",
        "agenda": "-",
        "time": "1pm",
        "date": "the 4th",
        "party": "sister",
        "event": "tennis activity"
      },
      {
        "room": "conference room 50",
        "agenda": "discuss the company picnic",
        "time": "6pm",
        "date": "the 8th",
        "party": "boss",
        "event": "meeting"
      },
      {
        "room": "-",
        "agenda": "-",
        "time": "7pm",
        "date": "friday",
        "party": "Jon",
        "event": "optometrist appointment"
      },
      {
        "room": "100 conference room",
        "agenda": "onboard new members",
        "time": "6pm",
        "date": "tuesday",
        "party": "management",
        "event": "conference"
      },
      {
        "room": "-",
        "agenda": "-",
        "time": "9am",
        "date": "the 10th",
        "party": "father",
        "event": "taking medicine"
      },
      {
        "room": "-",
        "agenda": "-",
        "time": "2pm",
        "date": "the 4th",
        "party": "aunt",
        "event": "doctor appointment"
      }
    ],
    "column_names": [
      "event",
      "time",
      "date",
      "room",
      "agenda",
      "party"
    ],
    "kb_title": "calendar"
  }
kbs = {'navigate': navigate, 'schedule': schedule, 'weather': weather}

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
    setattr(aeed_config, 'max_ctx_len', 16)  # kbs中items数量 + backward_size

    setattr(fsdg_config, 'data_dir', config.data_dir)
    setattr(fsdg_config, 'fsdg_domain_description', config.fsdg_domain_description)
    setattr(fsdg_config, 'laed_z_size', 10)
    setattr(fsdg_config, 'load_sess', config.load_sess)
    setattr(fsdg_config, 'use_gpu', config.use_gpu)
    setattr(fsdg_config, 'vocab', 'vocabs/stanford.json')
    setattr(fsdg_config, 'batch_size', 1)
    setattr(fsdg_config, 'rnn_cell', 'lstm')
    setattr(fsdg_config, 'embed_size', 200)
    setattr(fsdg_config, 'utt_type', 'rnn')
    setattr(fsdg_config, 'utt_cell_size', 256)
    setattr(fsdg_config, 'ctx_cell_size', 512)
    setattr(fsdg_config, 'dec_cell_size', 512)
    setattr(fsdg_config, 'bi_ctx_cell', False)
    setattr(fsdg_config, 'max_utt_len', 20)
    setattr(fsdg_config, 'max_dec_len', 40)
    setattr(fsdg_config, 'num_layer', 1)
    setattr(fsdg_config, 'use_attn', True)
    setattr(fsdg_config, 'attn_type', 'cat')
    setattr(fsdg_config, 'max_vocab_cnt', 10000)
    setattr(fsdg_config, 'op', 'adam')
    setattr(fsdg_config, 'backward_size', 14)
    setattr(fsdg_config, 'step_size', 2)
    setattr(fsdg_config, 'grad_clip', 3.0)
    setattr(fsdg_config, 'init_w', 0.08)
    setattr(fsdg_config, 'init_lr', 0.001)
    setattr(fsdg_config, 'momentum', 0.0)
    setattr(fsdg_config, 'lr_hold', 1)
    setattr(fsdg_config, 'lr_decay', 0.6)
    setattr(fsdg_config, 'dropout', 0.3)
    setattr(fsdg_config, 'improve_threshold', 0.996)
    setattr(fsdg_config, 'patient_increase', 2.0)
    setattr(fsdg_config, 'early_stop', True)
    setattr(fsdg_config, 'max_epoch', 50)
    setattr(fsdg_config, 'preview_batch_num', 50)
    setattr(fsdg_config, 'include_domain', True)
    setattr(fsdg_config, 'include_example', False)
    setattr(fsdg_config, 'include_state', True)
    setattr(fsdg_config, 'random_seed', config.random_seed)
    setattr(fsdg_config, 'dd_loss_coef', 0.0)
    setattr(fsdg_config, 'save_model', True)
    setattr(fsdg_config, 'print_step', 100)
    setattr(fsdg_config, 'ckpt_step', 400)
    setattr(fsdg_config, 'gen_type', 'greedy')
    setattr(fsdg_config, 'avg_type', 'word')
    setattr(fsdg_config, 'black_domains', 20)
    setattr(fsdg_config, 'beam_size', ['navigate'])
    setattr(fsdg_config, 'black_ratio', 1.0)
    setattr(fsdg_config, 'target_example_cnt', 150)
    setattr(fsdg_config, 'source_example_cnt', 150)
    setattr(fsdg_config, 'domain_description', 'nlu')
    setattr(fsdg_config, 'entities_file', 'NeuralDialog_ZSDG/data/stanford/kvret_entities.json')
    setattr(fsdg_config, 'action_match', True)
    setattr(fsdg_config, 'use_ptr', True)
    setattr(fsdg_config, 'forward_only', False)
    setattr(fsdg_config, 'max_ctx_len', 16)  # kbs中items数量 + backward_size

    aeed_corpus_client = getattr(corpora, aeed_config.corpus_client)(aeed_config)
    # print(aeed_corpus_client)
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
    is_kb = True
    while True:
        is_kb = input('whether use kb information: 1: yes, 2: no\n').strip()
        if is_kb == '1':
            is_kb = True
            print('kb information:\n', kbs[intent], '\n')
            break
        elif is_kb == '2':
            is_kb = False
            break
        else:
            print('Wrong choice')
    dio_turn = 0
    while True:
        dio_turn += 1
        usr_utt = input('Input(say \'thank you\'to terminate the conversation): ')
        if 'thank you' in usr_utt or 'thanks' in usr_utt:
            print('bot: ', 'you are welcome!')
            break
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

        kb_useful = kbs[intent]
        kb_useless = {}
        items = []
        column_names = []
        kb_title = 'unknown'
        kb_useless['items'] = items
        kb_useless['column_names'] = column_names
        kb_useless['kb_title'] = kb_title
        kb = kb_useful if is_kb else kb_useless

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
        # feature = np.array([1, 4, 5, 5, 8, 5, 5, 9, 9, 3]) if dio_turn==1 else np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        features.append(feature)
        # print('features: ', features)
        # print(utts)
        assert len(features) == len(utts)+1

        # TODO: get response from fsdg model
        # 将features加载到数据中

        fsdg_corpus_client.load_laed_z([[features]])

        fsdg_corpus_client.add_test_data(totol_data, features)

        test_dial = fsdg_corpus_client.get_corpus()['test']
        # print('test_dial', test_dial)

        fsdg_feed = fsdg_data_loaders.ZslLASMDDialDataLoaderPre("Test", test_dial, fsdg_config)

        response = hred_utils.predict(fsdg_model, fsdg_feed, fsdg_config, evaluator, num_batch=None)

        response = ' '.join(response.split()[2:])
        def clear_repeat(text):
            POUN = [',', '.', '?', '!']
            res = ['']
            for word in text.split():
                if word[-1] in POUN:
                    if word[:-1] == res[-1]:
                        res[-1] = word
                        continue
                else:
                    if word == res[-1]:
                        res[-1] = word
                        continue
                res.append(word)
            return ' '.join(res)
        response = clear_repeat(response)
        print('bot:', response)
        if 'you are welcome' in response or 'you\' re welcome' in response:
            break
        utts.append(response)
        features.append(np.array([0]*10))


def run_demo():
    # Data
    data_arg = add_argument_group('Data')
    data_arg.add_argument('--model', default='logs/2021-06-10T19-32-51-ae_ed.py')
    data_arg.add_argument('--load_sess', default='logs/2021-06-09T19-31-12-train_fsdg.py-6a61fab0')
    data_arg.add_argument('--model_name', default='AeED')
    data_arg.add_argument('--model_type', default='dialog')
    data_arg.add_argument('--data_dir', nargs='+', default=['NeuralDialog_ZSDG/data/stanford'])
    data_arg.add_argument('--corpus_client', default='ZslStanfordCorpusPre')
    data_arg.add_argument('--data_loader', default='SMDDataLoaderPre')
    data_arg.add_argument('--batch_size', default=1)
    data_arg.add_argument('--vocab', default=None)
    data_arg.add_argument('--domain_description', default='annotation')
    data_arg.add_argument('--fsdg_domain_description', default='nlu')

    # MISC
    misc_arg = add_argument_group('Misc')
    misc_arg.add_argument('--use_gpu', type=str2bool, default=False)
    misc_arg.add_argument('--forward_only', type=str2bool, default=True)
    misc_arg.add_argument('--gen_type', type=str, default='greedy')
    misc_arg.add_argument('--random_seed', type=int, default=2021)

    config, unparsed = get_config()
    fix_random_seed(config.random_seed)
    main(config)

if __name__ == "__main__":
    run_demo()
