import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pickle
import json
import numpy as np


def draw_training_curve(args):
    record = pickle.load(open(os.path.join(args.save_path, 'checkpoint', 'record.pickle'), 'rb'))

    epochs = record['epochs']
    loss_train = {i:[] for i in record['record_name']}
    loss_test = {i:[] for i in record['record_name']}

    for name in record['record_name']:
        for epoch in range(1, epochs+1):
            loss_train[name].append(record['train'][epoch][name])
            loss_test[name].append(record['test'][epoch][name])

    for name in record['record_name']:
        fig = plt.figure()
        plt.title(name)
        plt.plot(loss_train[name], label='train')
        plt.plot(loss_test[name], label='test')
        plt.legend()
        fig.savefig(os.path.join(args.save_path, 'checkpoint', 'loss_curve_{}.jpg'.format(name)))


def draw_actionness(args, mode, epoch):

    results = pickle.load(open(os.path.join(args.save_path, 'predicts', '{}_results_epoch{}.pickle'.format(mode, epoch)), 'rb'))

    if not os.path.exists(os.path.join(args.save_path, 'actionness', mode, 'epoch{}'.format(epoch))):
        os.makedirs(os.path.join(args.save_path, 'actionness', mode, 'epoch{}'.format(epoch)))

    for key, data in results.items():

        fig = plt.figure()
        plt.subplot(311)
        plt.title('action')
        plt.plot(data['gt_action'])
        plt.plot(data['action_heat'])
        plt.subplot(312)
        plt.title('start')
        plt.plot(data['gt_start'])
        plt.plot(data['start_heat'])
        plt.subplot(313)
        plt.title('end')
        plt.plot(data['gt_end'])
        plt.plot(data['end_heat'])
        fig.savefig(os.path.join(args.save_path, 'actionness', mode, 'epoch{}'.format(epoch), '{}.jpg'.format(key)))
        plt.close(fig)



def draw_proposals(args, mode, epoch):
    picklefile = os.path.join(args.save_path, 'predicts', '{}_results_epoch{}.pickle'.format(mode, epoch))
    resultfile = os.path.join(args.save_path, 'proposals', 'results_epoch{}.json'.format(epoch))

    gt = pickle.load(open(picklefile, 'rb'))
    results = json.load(open(resultfile))

    if not os.path.exists(os.path.join(args.save_path, 'plotproposals', mode, 'epoch{}'.format(epoch))):
        os.makedirs(os.path.join(args.save_path, 'plotproposals', mode, 'epoch{}'.format(epoch)))

    for key, data in gt.items():

        segs = results['results'][key]
        # proposals = np.zeros(len(data['gt_action']))

        # for seg in segs:
        #     proposals[int(seg['segment'][0]/0.64):int(seg['segment'][1]/0.64)] = seg['score']

        fig = plt.figure()

        plt.title('proposals & GT')

        plt.plot(data['gt_action'])

        for seg in segs:
            # proposals[int(seg['segment'][0]/0.64):int(seg['segment'][1]/0.64)] = seg['score']
            plt.plot([seg['segment'][0]/0.64, seg['segment'][1]/0.64],[seg['score'], seg['score']])

        fig.savefig(os.path.join(args.save_path, 'plotproposals', mode, 'epoch{}'.format(epoch), '{}.jpg'.format(key)))



