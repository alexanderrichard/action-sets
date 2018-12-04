#!/usr/bin/python2.7

import numpy as np
import scipy.optimize
import random
import argparse
import multiprocessing as mp
import queue
from utils.dataset import Dataset
from utils.network import Trainer, Forwarder
from utils.grammar import PathGrammar
from utils.length_model import PoissonModel
from utils.viterbi import Viterbi


### prior ######################################################################
def estimate_prior(dataset):
    prior = np.zeros( (dataset.n_classes,), dtype=np.float32 )
    for video in dataset.videos():
        for c in range(dataset.n_classes):
            prior[c] += dataset.features[video].shape[1] if c in dataset.action_set[video] else 0
    return prior / np.sum(prior)


### loss based lengths #########################################################
def loss_based_lengths(dataset):
    # definition of objective function
    def objective(x, A, l):
        return 0.5 * np.sum( (np.dot(A, x) - l) ** 2 )
    # number of frames per video
    vid_lengths = np.array( [dataset.length(video) for video in dataset.videos()] )
    # binary data matrix: (n_videos x n_classes), A[video, class] = 1 iff class in action_set[video]
    A = np.zeros((len(dataset.videos()), dataset.n_classes))
    for i, video in enumerate(dataset.videos()):
        for c in dataset.action_set[video]:
           A[i, c] = 1
    # constraints: each mean length is at least 50 frames
    constr = [ lambda x, i=i : x[i] - 50 for i in range(dataset.n_classes) ]
    # optimize
    x0 = np.ones((dataset.n_classes)) * 450.0 # some initial value
    mean_lengths = scipy.optimize.fmin_cobyla(objective, x0, constr, args=(A, vid_lengths), consargs=(), maxfun=10000, disp=False)
    return mean_lengths


### monte-carlo grammar ########################################################
def monte_carlo_grammar(dataset, mean_lengths, index2label, max_paths = 1000):
    monte_carlo_grammar = []
    sil_length = mean_lengths[0]
    while len(monte_carlo_grammar) < max_paths:
        for video in dataset.videos():
            action_set = dataset.action_set[video] - set([0]) # exclude SIL
            seq = []
            while sum( [ mean_lengths[label] for label in seq ] ) + 2 * sil_length < dataset.length(video):
                seq.append( random.choice(list(action_set)) )
            if len(seq) == 0: # omit empty sequences
                continue
            monte_carlo_grammar.append('SIL ' + ' '.join( [index2label[idx] for idx in seq] ) + ' SIL')
    random.shuffle(monte_carlo_grammar)
    return monte_carlo_grammar[0:max_paths]


################################################################################
### TRAINING                                                                 ###
################################################################################
def train(label2index, index2label):
    # list of train videos
    with open('data/split1.train', 'r') as f:
        video_list = f.read().split('\n')[0:-1]
    # read train set
    print('read data...')
    dataset = Dataset('data', video_list, label2index)
    print('done')
    # train the network
    trainer = Trainer(dataset)
    trainer.train(batch_size = 512, n_epochs = 2, learning_rate = 0.1)
    trainer.save_model('results/net.model')
    # estimate prior, loss-based lengths, and monte-carlo grammar
    prior = estimate_prior(dataset)
    mean_lengths = loss_based_lengths(dataset)
    grammar = monte_carlo_grammar(dataset, mean_lengths, index2label)
    np.savetxt('results/prior', prior)
    np.savetxt('results/mean_lengths', mean_lengths, fmt='%.3f')
    with open('results/grammar', 'w') as f:
        f.write('\n'.join(grammar) + '\n')


################################################################################
### INFERENCE                                                                ###
################################################################################
def infer(label2index, index2label, n_threads):
    # load models
    log_prior = np.log( np.loadtxt('results/prior') )
    grammar = PathGrammar('results/grammar', label2index)
    length_model = PoissonModel('results/mean_lengths', max_length = 2000)
    forwarder = Forwarder('results/net.model')
    # Viterbi decoder (max_hypotheses = n: at each time step, prune all hypotheses worse than the top n)
    viterbi_decoder = Viterbi(grammar, length_model, frame_sampling = 30, max_hypotheses = 50000 )
    # create list of test videos
    with open('data/split1.test', 'r') as f:
        video_list = f.read().split('\n')[0:-1]
    # forward each video
    log_probs = dict()
    queue = mp.Queue()
    for video in video_list:
        queue.put(video)
        dataset = Dataset('data', [video], label2index)
        log_probs[video] = forwarder.forward(dataset) - log_prior
        log_probs[video] = log_probs[video] - np.max(log_probs[video])
    # Viterbi decoding
    procs = []
    for i in range(n_threads):
        p = mp.Process(target = decode, args = (queue, log_probs, viterbi_decoder, index2label) )
        procs.append(p)
        p.start()
    for p in procs:
        p.join()


### helper function for parallelized Viterbi decoding ##########################
def decode(queue, log_probs, decoder, index2label):
    while not queue.empty():
        try:
            video = queue.get(timeout = 3)
            score, labels, segments = decoder.decode( log_probs[video] )
            # save result
            with open('results/' + video, 'w') as f:
                f.write( '### Recognized sequence: ###\n' )
                f.write( ' '.join( [index2label[s.label] for s in segments] ) + '\n' )
                f.write( '### Score: ###\n' + str(score) + '\n')
                f.write( '### Frame level recognition: ###\n')
                f.write( ' '.join( [index2label[l] for l in labels] ) + '\n' )
        except queue.Empty:
            pass



################################################################################
### MAIN                                                                     ###
################################################################################
if __name__ == '__main__':

    # read label2index mapping and index2label mapping
    label2index = dict()
    index2label = dict()
    with open('data/mapping.txt', 'r') as f:
        content = f.read().split('\n')[0:-1]
        for line in content:
            label2index[line.split()[1]] = int(line.split()[0])
            index2label[int(line.split()[0])] = line.split()[1]

    ### command line arguments ###
    ### mode: either training or inference
    ### --n_threads: number of threads to use for inference (not used in training mode)
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices = ['training', 'inference'])
    parser.add_argument('--n_threads', default = 1, type = int)
    args = parser.parse_args()

    if args.mode == 'training':
        train(label2index, index2label)
    else:
        infer(label2index, index2label, args.n_threads)

