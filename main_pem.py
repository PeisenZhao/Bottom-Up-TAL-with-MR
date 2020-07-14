import tensorflow as tf
import time, os, sys
from tqdm import tqdm
import math
import numpy as np
import pickle, json
from scipy.interpolate import interp1d

from model import pemNet
from dataloader import pem_DataLoader
from generate_proposals import generate_proposals
from eval import evaluation_proposal
from configuration import Logger, parse_base_args



def iou_score(gt, anchor):
    gt_min, gt_max = gt
    an_min, an_max = anchor
    if (an_min >= gt_max) or (gt_min >= an_max):
        return 0.
    else:
        union = max(gt_max, an_max) - min(gt_min, an_min)
        inter = min(gt_max, an_max) - max(gt_min, an_min)
        return float(inter) / union


def run_training(args):
    with tf.Graph().as_default():

        global_step = tf.Variable(0, trainable=False)

        # build model
        model_input = tf.placeholder(tf.float32, shape=[args.pem_batch,96])
        iou = tf.placeholder(tf.float32, shape=[args.pem_batch,1])

        model = pemNet(model_input, iou)

        saver = tf.train.Saver(max_to_keep=args.pem_epochs)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        record = {'test':{}, 'train':{}, 'epochs':args.pem_epochs, 'record_name':[i for i in model.loss.keys()]}

        with tf.Session(config=config) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            summary_writer = tf.summary.FileWriter(os.path.join(args.save_path, 'checkpoint/'), graph=sess.graph)
            DL = pem_DataLoader(batch=args.pem_batch, shuffle=True, datafile=os.path.join(args.save_path, 'proposals/pem_data_n5_epoch10.pickle'))
            for epoch in range(args.pem_epochs):
                print('-----------Epoch {} -------------'.format(epoch+1))
                # training
                if epoch < 30:
                    lr = [1e-3]
                else:
                    lr = [1e-4]
                print ('Start training, total batches in train set is: %d'%(DL.train_nbatch))
                loss_name = [i for i in model.loss.keys()]
                loss_tensor = [i for i in model.loss.values()]
                loss_record = {i:[] for i in loss_name}
                with tqdm (total=DL.train_nbatch) as count:
                    for step in range(DL.train_nbatch):
                        ff, ii = DL.generate_batch('train', step)
                        values = sess.run(loss_tensor+[model.solver], feed_dict={model_input:ff, iou:ii, model.lr:lr})
                        for i, v in enumerate(values):
                            if i < len(loss_name):
                                loss_record[loss_name[i]].append(v)
                            # if i > len(loss_name):
                            #     print(v)
                        count.update(1)
                print('Training results:')
                for name, value in loss_record.items():
                    loss_record[name] = np.mean(value)
                    print(name, np.mean(value))
                record['train'][epoch+1] = loss_record

                # testing and save
                if (epoch+1) % 1 == 0:

                    print('Start testing, total batches in test set is: %d'%(DL.val_nbatch))
                    loss_name = [i for i in model.loss.keys()]
                    loss_tensor = [i for i in model.loss.values()]
                    loss_record = {i:[] for i in loss_name}

                    with tqdm (total=DL.val_nbatch) as count:
                        for step in range(DL.val_nbatch):
                            ff, ii = DL.generate_batch('val', step)
                            values = sess.run(loss_tensor, feed_dict={model_input:ff, iou:ii})
                            for i, v in enumerate(values):
                                if i < len(loss_name):
                                    loss_record[loss_name[i]].append(v)
                            count.update(1)
                    print('Testing results:')
                    for name, value in loss_record.items():
                        loss_record[name] = np.mean(value)
                        print(name, np.mean(value))
                    record['test'][epoch+1] = loss_record

                    checkpoint_path = os.path.join(args.save_path, 'pem_checkpoint', 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=(epoch+1))

        with open(os.path.join(args.save_path, 'pem_checkpoint', 'record.pickle'), 'wb') as f:
            pickle.dump(record, f)


def generate_proposals_with_pem(args, epoch, mode):

    # predicted action start end heat
    picklefile = os.path.join(args.save_path, 'predicts', '{}_results_epoch{}.pickle'.format(mode, epoch))
    results = pickle.load(open(picklefile, 'rb'))


    # pem model

    with tf.Graph().as_default():
        model_input = tf.placeholder(tf.float32, shape=[None,96])
        gt_iou = tf.placeholder(tf.float32, shape=[None,1])
        model = pemNet(model_input, gt_iou)

        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver.restore(sess, os.path.join(args.save_path, 'pem_checkpoint', 'model.ckpt-{}'.format(40)))


            # results data
            proposalfile = os.path.join(args.save_path, 'proposals', 'results_softnms_n5_score_se_epoch{}.json'.format(epoch))
            data = json.load(open(proposalfile))

            data_path = args.data_path
            anno_path = os.path.join(data_path, 'annotation')
            alldata = json.load(open(os.path.join(anno_path, 'thumos14.json')))['database']

            with tqdm (total=len(data['results'].keys())) as count:

                for key, item in data['results'].items():

                    t_step = args.t_step / args.fps[key]
                    t_granularity = args.t_granularity / args.fps[key]
                    length = int((alldata[key]['fealength_step4']+args.down_sample-1) / args.down_sample)
                    t_length = t_granularity/2. + t_step*(length-1)

                    granularity_list = [t_granularity/2. + t_step*(l) for l in range(length)]

                    most_small = granularity_list[0]+1e-2
                    most_large = granularity_list[-1]-1e-2

                    predictheat = results[key]

                    af = interp1d(granularity_list, predictheat['action_heat'][:length])
                    sf = interp1d(granularity_list, predictheat['start_heat'][:length])
                    ef = interp1d(granularity_list, predictheat['end_heat'][:length])

                    for it in item:
                        ps = it['segment'][0]
                        pe = it['segment'][1]
                        duration = pe - ps
                        iou_list = []
                        for gt in alldata[key]['annotations']:
                            # print([ps,pe], gt['segment'])

                            iou_list.append(iou_score([ps,pe], gt['segment']))
                        iou = max(iou_list)

                        ps_new = min(max(ps-0.2*duration, most_small), most_large)
                        pe_new = max(min(most_large, pe+0.2*duration), most_small)

                        step = (pe_new - ps_new)/31
                        index = [ps_new + step*i for i in range(32)]
                        # print(index[-1], granularity_list[-1], '---', index[0], granularity_list[0])
                        pem_fea = np.expand_dims(np.hstack([sf(index),af(index),ef(index)]), 0)

                        pre_score = sess.run(model.output, feed_dict={model_input:pem_fea})

                        it['score'] = float(it['score'] * pre_score)
                        # it['score'] = iou
                        it['oracle'] = iou

                    count.update(1)

    if not os.path.exists(os.path.join(args.save_path, 'proposals')):
        os.makedirs(os.path.join(args.save_path, 'proposals'))

    with open(os.path.join(args.save_path, 'proposals', 'results_softnms_n5_score_se_pem_epoch{}.json'.format(epoch)), 'w') as f:
        json.dump(data, f)




if __name__ == '__main__':

    args = parse_base_args()

    # constraint GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # set log file
    sys.stdout = Logger(os.path.join(args.save_path, 'log_pem.log'))

    # step 0
    generate_proposals(args, 10, 'validation', prepare_pemdata='True')

    # step1 train models
    run_training(args)

    # step2 generate final proposal
    for i in range(1, args.epochs+1):
        generate_proposals_with_pem(args, i, 'test')

    # step3 evaluation
    AR_AN = {}
    aran = []
    for epoch in range(1, args.epochs+1):
        eval_file = os.path.join(args.save_path, 'proposals', 'results_softnms_n5_score_se_pem_epoch{}.json'.format(epoch))
        results = evaluation_proposal(args, eval_file)
        AR_AN[epoch] = results
        aran.append(results[1])
    aran = np.array(aran)
    mean_10 = np.mean(aran[10:,:],0)
    with open(os.path.join(args.save_path, 'result_softnms_n5_score_se_pem.txt'), 'w') as f:
        for key, item in AR_AN.items():
            s = '{}\t{}\t{}\t{}\t{}\t{}\n'.format(key,item[1][0],item[1][1],item[1][2],item[1][3],item[1][4])
            f.write(s)
        s = '{}\t{}\t{}\t{}\t{}\t{}\n'.format('averaged_last10',mean_10[0],mean_10[1],mean_10[2],mean_10[3],mean_10[4])
        f.write(s)


