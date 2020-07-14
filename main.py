import tensorflow as tf
import time, os, sys
from tqdm import tqdm
import math
import numpy as np
import pickle

from model import MultiDenseNet
from dataloader import DataLoader
from plot import draw_training_curve, draw_actionness, draw_proposals
from configuration import Logger, parse_base_args
from generate_proposals import generate_proposals
from eval import evaluation_proposal




def run_training(args):
    with tf.Graph().as_default():

        global_step = tf.Variable(0, trainable=False)

        # build model
        model_input = tf.placeholder(tf.float32, shape=[args.batch,args.in_window,2048])
        anno_action = tf.placeholder(tf.float32, shape=[args.batch,args.out_window,1])
        anno_point = tf.placeholder(tf.float32, shape=[args.batch,2,args.out_window,1])
        anno_bias = tf.placeholder(tf.float32, shape=[args.batch,2,args.out_window,1])
        anno_mask = tf.placeholder(tf.float32, shape=[args.batch,args.out_window,1])

        model = MultiDenseNet(args, model_input, anno_action, anno_point, anno_bias, anno_mask)

        saver = tf.train.Saver(max_to_keep=args.epochs)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        record = {'test':{}, 'train':{}, 'epochs':args.epochs, 'record_name':[i for i in model.loss.keys()]}

        with tf.Session(config=config) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            summary_writer = tf.summary.FileWriter(os.path.join(args.save_path, 'checkpoint/'), graph=sess.graph)
            start_time = time.time()
            for epoch in range(args.epochs):
                print('-----------Epoch {} -------------'.format(epoch+1))
                # training

                if epoch < 10:
                    lr = [1e-3]
                else:
                    lr = [1e-4]


                DL = DataLoader(args=args, mode='validation', batch=args.batch, shuffle=True)
                print('Start training, total batches in train set is: %d'%(DL.nbatch))
                loss_name = [i for i in model.loss.keys()]
                loss_tensor = [i for i in model.loss.values()]
                loss_record = {i:[] for i in loss_name}

                with tqdm (total=DL.nbatch) as count:
                    for step in range(DL.nbatch):
                        aa, pp, bb, ff, mm = DL.gen_train_batch(step)
                        values = sess.run(loss_tensor+[model.solver], feed_dict={model_input:ff, anno_action:aa, anno_point:pp, anno_bias:bb, anno_mask:mm, model.lr:lr})
                        for i, v in enumerate(values):
                            if i < len(loss_name):
                                loss_record[loss_name[i]].append(v)


                        count.update(1)

                print('Training results:')
                for name, value in loss_record.items():
                    loss_record[name] = np.mean(value)
                    print(name, np.mean(value))

                record['train'][epoch+1] = loss_record

                # testing and save
                if (epoch+1) % 1 == 0:

                    DL = DataLoader(args=args, mode='test', batch=args.batch, shuffle=False)
                    print('Start testing, total batches in test set is: %d'%(DL.nbatch))
                    loss_name = [i for i in model.loss.keys()]
                    loss_tensor = [i for i in model.loss.values()]
                    heat_name = ['action_heat', 'start_heat', 'end_heat']
                    heat_tensor = [model.action_heat, model.start_heat, model.end_heat]
                    gt_name = ['gt_action', 'gt_start', 'gt_end']
                    gt_tensor = [model.gt_action, model.gt_point[:,0,:,:], model.gt_point[:,1,:,:]]
                    loss_record = {i:[] for i in loss_name}
                    heat_record = {i:[] for i in heat_name}
                    gt_record = {i:[] for i in gt_name}

                    with tqdm (total=DL.nbatch) as count:
                        for step in range(DL.nbatch):
                            aa, pp, bb, ff, mm = DL.gen_train_batch(step)
                            values = sess.run(loss_tensor+heat_tensor+gt_tensor, feed_dict={model_input:ff, anno_action:aa, anno_point:pp, anno_bias:bb, anno_mask:mm})
                            for i, v in enumerate(values):
                                if i < len(loss_name):
                                    loss_record[loss_name[i]].append(v)
                                elif i < len(loss_name) + len(heat_name):
                                    heat_record[heat_name[i-len(loss_name)]].append(v)
                                else:
                                    gt_record[gt_name[i-len(loss_name)-len(heat_name)]].append(v)
                            count.update(1)

                    print('Testing results:')
                    for name, value in loss_record.items():
                        loss_record[name] = np.mean(value)
                        print(name, np.mean(value))

                    record['test'][epoch+1] = loss_record

                    checkpoint_path = os.path.join(args.save_path, 'checkpoint', 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=(epoch+1))

        with open(os.path.join(args.save_path, 'checkpoint', 'record.pickle'), 'wb') as f:
            pickle.dump(record, f)



def run_evaluating(args, epoch, mode):

    with tf.Graph().as_default():

        # build model
        model_input = tf.placeholder(tf.float32, shape=[1,None,2048])
        anno_action = tf.placeholder(tf.float32, shape=[1,None,1])
        anno_point = tf.placeholder(tf.float32, shape=[1,2,None,1])
        anno_bias = tf.placeholder(tf.float32, shape=[1,2,None,1])
        anno_mask = tf.placeholder(tf.float32, shape=[1,None,1])

        model = MultiDenseNet(args, model_input, anno_action, anno_point, anno_bias, anno_mask)

        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
          
            saver.restore(sess, os.path.join(args.save_path, 'checkpoint', 'model.ckpt-{}'.format(epoch)))

            DL = DataLoader(args=args, mode=mode, batch=1, shuffle=False)

            results = {}
            out_names = ['action_heat', 'start_heat', 'end_heat', 'gt_action', 'gt_start', 'gt_end', 'start_regr', 'end_regr']
            out_tensors = [model.action_heat, model.start_heat, model.end_heat, model.gt_action, model.gt_point[:,0,:,:], model.gt_point[:,1,:,:], model.start_bias, model.end_bias]

            with tqdm (total=DL.size) as count:

                for video in range(DL.size):

                    key, aa, pp, bb, ff, mm = DL.gen_eval_batch(video)
                    values = sess.run(out_tensors, feed_dict={model_input:ff, anno_action:aa, anno_point:pp, anno_bias:bb, anno_mask:mm})

                    out_record = {}
                    for i, v in enumerate(values):
                        out_record[out_names[i]] = np.squeeze(v)
                    results[key] = out_record

                    count.update(1)

        if not os.path.exists(os.path.join(args.save_path, 'predicts')):
            os.makedirs(os.path.join(args.save_path, 'predicts'))
        with open(os.path.join(args.save_path, 'predicts', '{}_results_epoch{}.pickle'.format(mode, epoch)), 'wb') as f:
            pickle.dump(results, f)



if __name__ == '__main__':

    args = parse_base_args()

    # constraint GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # set log file
    sys.stdout = Logger(os.path.join(args.save_path, 'log.log'))

    # step1 train models
    run_training(args)

    # step2 evaluate models with different epoch
    for epoch in range(1, args.epochs+1):
        # test
        run_evaluating(args, epoch, 'test')
        # train
        if epoch == 10:
            run_evaluating(args, epoch, 'validation')

    # step3 draw loss curve
    draw_training_curve(args)

    # step4 generate proposals
    for epoch in range(1, args.epochs+1):    	
        generate_proposals(args, epoch, 'test', prepare_pemdata=False)

    # step5 evaluation

    AR_AN = {}
    aran = []

    for epoch in range(1, args.epochs+1):
        eval_file = os.path.join(args.save_path, 'proposals', 'results_softnms_n5_score_se_epoch{}.json'.format(epoch))
        results = evaluation_proposal(args, eval_file)
        AR_AN[epoch] = results
        aran.append(results[1])
    aran = np.array(aran)
    mean_10 = np.mean(aran[10:,:],0)

    with open(os.path.join(args.save_path, 'result_softnms_n5_score_se.txt'), 'w') as f:
        for key, item in AR_AN.items():
            s = '{}\t{}\t{}\t{}\t{}\t{}\n'.format(key,item[1][0],item[1][1],item[1][2],item[1][3],item[1][4])
            f.write(s)
        s = '{}\t{}\t{}\t{}\t{}\t{}\n'.format('averaged_last10',mean_10[0],mean_10[1],mean_10[2],mean_10[3],mean_10[4])
        f.write(s)

    # step6 plot actionness and proposals

    # epoch = 20
    # draw_actionness('test', epoch)
    # draw_proposals('test', epoch)

