from __future__ import print_function

import argparse
import json
import logging
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), 'NeuralDialog_LAED'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'NeuralDialog_ZSDG'))

from NeuralDialog_LAED.laed import evaluators, utt_utils, dialog_utils
from NeuralDialog_LAED.laed import main as engine
from NeuralDialog_LAED.laed.dataset import corpora
from NeuralDialog_LAED.laed.dataset import data_loaders
from NeuralDialog_LAED.laed.models import dialog_models
from NeuralDialog_LAED.laed.utils import str2bool, prepare_dirs_loggers, get_time, process_config

from utils import corpora
from utils.corpora import load_vocab

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('data_dir', nargs='+')
data_arg.add_argument('corpus_client', help='LAEDBlisCorpus/StanfordCorpus')
data_arg.add_argument('--vocab', type=str, default='none')
data_arg.add_argument('--exclude_domains', nargs='*', default=[])
data_arg.add_argument('--log_dir', type=str, default='logs')

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--use_reg_kl', type=str2bool, default=True)
net_arg.add_argument('--use_attribute', type=str2bool, default=True)
net_arg.add_argument('--y_size', type=int, default=3)
net_arg.add_argument('--k', type=int, default=5)

net_arg.add_argument('--rnn_cell', type=str, default='gru')
net_arg.add_argument('--embed_size', type=int, default=200)
net_arg.add_argument('--utt_type', type=str, default='rnn')
net_arg.add_argument('--utt_cell_size', type=int, default=256)
net_arg.add_argument('--ctx_cell_size', type=int, default=512)
net_arg.add_argument('--dec_cell_size', type=int, default=512)
net_arg.add_argument('--bi_ctx_cell', type=str2bool, default=False)
net_arg.add_argument('--max_utt_len', type=int, default=40)
net_arg.add_argument('--max_dec_len', type=int, default=40)
net_arg.add_argument('--max_vocab_cnt', type=int, default=10000)
net_arg.add_argument('--num_layer', type=int, default=1)
net_arg.add_argument('--use_attn', type=str2bool, default=False)
net_arg.add_argument('--attn_type', type=str, default='cat')
net_arg.add_argument('--greedy_q', type=str2bool, default=True)

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--op', type=str, default='adam')
train_arg.add_argument('--backward_size', type=int, default=10)
train_arg.add_argument('--step_size', type=int, default=1)
train_arg.add_argument('--grad_clip', type=float, default=3.0)
train_arg.add_argument('--init_w', type=float, default=0.08)
train_arg.add_argument('--init_lr', type=float, default=0.001)
train_arg.add_argument('--momentum', type=float, default=0.0)
train_arg.add_argument('--lr_hold', type=int, default=1)
train_arg.add_argument('--lr_decay', type=float, default=0.6)
train_arg.add_argument('--dropout', type=float, default=0.3)
train_arg.add_argument('--improve_threshold', type=float, default=0.996)
train_arg.add_argument('--patient_increase', type=float, default=2.0)
train_arg.add_argument('--early_stop', type=str2bool, default=True)
train_arg.add_argument('--max_epoch', type=int, default=1000)
train_arg.add_argument('--loss_type', type=str, default="e2e")
train_arg.add_argument('--include_eod', type=str2bool, default=True)
train_arg.add_argument('--include_domain', type=str2bool, default=True)

# MISC
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--save_model', type=str2bool, default=True)
misc_arg.add_argument('--use_gpu', type=str2bool, default=True)
misc_arg.add_argument('--fix_batch', type=str2bool, default=False)
misc_arg.add_argument('--print_step', type=int, default=500)
misc_arg.add_argument('--ckpt_step', type=int, default=1000)
misc_arg.add_argument('--freeze_step', type=int, default=5000)
misc_arg.add_argument('--batch_size', type=int, default=30)
misc_arg.add_argument('--preview_batch_num', type=int, default=1)
misc_arg.add_argument('--gen_type', type=str, default='greedy')
misc_arg.add_argument('--avg_type', type=str, default='word')
misc_arg.add_argument('--beam_size', type=int, default=10)
misc_arg.add_argument('--forward_only', type=str2bool, default=False)
data_arg.add_argument('--load_sess', type=str, default="2018-02-14T12-55-34-stanford-skip.py")
logger = logging.getLogger()


def main(config):
    corpus_client = getattr(corpora, config.corpus_client)(config)
    if config.vocab == 'none':
        corpus_name = config.data_dir[0].split('/')[-1]
        config.vocab = f'vocabs/{corpus_name}.json'
    corpus_client.vocab, corpus_client.rev_vocab, corpus_client.unk_id = load_vocab(config.vocab)
    prepare_dirs_loggers(config, os.path.basename(__file__))

    dial_corpus = corpus_client.get_corpus()
    train_dial, valid_dial, test_dial = (dial_corpus['train'],
                                         dial_corpus['valid'],
                                         dial_corpus['test'])

    evaluator = evaluators.BleuEvaluator('customer_service')

    # create data loader that feed the deep models
    train_feed = data_loaders.SMDDialogSkipLoader("Train", train_dial, config)
    valid_feed = data_loaders.SMDDialogSkipLoader("Valid", valid_dial, config)
    test_feed = data_loaders.SMDDialogSkipLoader("Test", test_dial, config)
    model = dialog_models.StED(corpus_client, config)

    if config.forward_only:
        test_file = os.path.join(config.log_dir, config.load_sess,
                                 "{}-test-{}.txt".format(get_time(), config.gen_type))
        dump_file = os.path.join(config.log_dir, config.load_sess,
                                 "{}-z.pkl".format(get_time()))
        model_file = os.path.join(config.log_dir, config.load_sess, "model")
    else:
        test_file = os.path.join(config.session_dir,
                                 "{}-test-{}.txt".format(get_time(), config.gen_type))
        dump_file = os.path.join(config.session_dir, "{}-z.pkl".format(get_time()))
        model_file = os.path.join(config.session_dir, "model")

    if config.use_gpu:
        model.cuda()

    if config.forward_only is False:
        try:
            engine.train(model,
                         train_feed,
                         valid_feed,
                         test_feed,
                         config,
                         evaluator,
                         gen=dialog_utils.generate_with_adv)
        except KeyboardInterrupt:
            print("Training stopped by keyboard.")

    config.batch_size = 10
    model.load_state_dict(torch.load(model_file))
    engine.validate(model, valid_feed, config)
    engine.validate(model, test_feed, config)

    dialog_utils.generate_with_adv(model, test_feed, config, None, num_batch=0)
    selected_clusters = utt_utils.latent_cluster(model, train_feed, config, num_batch=None)
    if selected_clusters:
        selected_outs = dialog_utils.selective_generate(model, test_feed, config, selected_clusters)
        print(len(selected_outs))

        with open(os.path.join(dump_file+'.json'), 'w') as f:
            json.dump(selected_clusters, f, indent=2)

        with open(os.path.join(dump_file+'.out.json'), 'w') as f:
            json.dump(selected_outs, f, indent=2)

    with open(os.path.join(dump_file), "wb") as f:
        print("Dumping test to {}".format(dump_file))
        dialog_utils.dump_latent(model, test_feed, config, f, num_batch=None)

    with open(os.path.join(test_file), "w") as f:
        print("Saving test to {}".format(test_file))
        dialog_utils.gen_with_cond(model, test_feed, config, num_batch=None,
                                   dest_f=f)
    with open(os.path.join(test_file+'.txt'), "w") as f:
        print("Saving test to {}".format(test_file))
        dialog_utils.generate(model, test_feed, config, evaluator, num_batch=None,
                                   dest_f=f)


if __name__ == "__main__":
    config, unparsed = get_config()
    config = process_config(config)
    main(config)
