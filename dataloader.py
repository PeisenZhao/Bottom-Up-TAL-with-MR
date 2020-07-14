import numpy as np
import pandas as pd
import json
import math
import random
import pickle
from tqdm import tqdm
import time
import os

class DataLoader():
    def __init__(self, args, mode, batch, shuffle):

        self.args = args

        self.data_path = self.args.data_path
        self.anno_path = os.path.join(self.data_path, 'annotation')

        self.mode = mode
        self.shuffle = shuffle
        self.batch = batch

        self.feature = os.path.join(self.data_path, 'I3D_features')
        self.data_segments = self.gen_dataset()
        self.size = len(self.data_segments)
        self.nbatch = int(self.size / self.batch)

    def gen_dataset(self):

        alldata = json.load(open(os.path.join(self.anno_path, 'thumos14.json')))['database']
        database = {}
        for video in alldata.keys():
            if alldata[video]['subset'] == self.mode:
                database[video] = alldata[video]

        data_segments = [] 
        for key, video in database.items():
            t_granularity = self.args.t_granularity/self.args.fps[key]
            t_step = self.args.t_step/self.args.fps[key]
            fealength = int((video['fealength_step4']+self.args.down_sample-1) / self.args.down_sample)
            actions = np.zeros([fealength, self.args.class_num])
            points = np.zeros([2, fealength, self.args.class_num])
            biases = np.zeros([2, fealength, self.args.class_num])
            annotation = video['annotations']
            for anno in annotation:
                # time unit: sec
                s0 = float(anno['segment'][0])
                e0 = float(anno['segment'][1])
                l = e0 - s0
                s1 = max(s0-l/10., 0.0)
                s2 = (s0+l/10.)
                e1 = (e0-l/10.)
                e2 = min(float((fealength-1)*t_step+(t_granularity/2.)), e0+l/10.)

                is0 = max(0, round((s0-t_granularity/2.)/t_step))
                is1 = max(0, round((s1-t_granularity/2.)/t_step))
                is2 = max(0, round((s2-t_granularity/2.)/t_step))
                ie0 = min((fealength-1), round((e0-t_granularity/2.)/t_step))
                ie1 = min((fealength-1), round((e1-t_granularity/2.)/t_step))
                ie2 = min((fealength-1), round((e2-t_granularity/2.)/t_step))


                actions[is0:ie0+1,anno['labelidx']] = 1
                points[0,is1:is2+1,anno['labelidx']] = 1
                points[1,ie1:ie2+1,anno['labelidx']] = 1

                if len(biases[0,is1:is2+1,anno['labelidx']]) != len(range(is1,is2+1)) or len(biases[1,ie1:ie2+1,anno['labelidx']]) != len(range(ie1,ie2+1)):
                    # print(key,anno['labelidx'],fealength, is1,is2+1,ie1,ie2+1)
                    continue
                else:
                    biases[0,is1:is2+1,anno['labelidx']] = [s0 - (t*t_step+t_granularity/2.) for t in range(is1,is2+1)] 
                    biases[1,ie1:ie2+1,anno['labelidx']] = [e0 - (t*t_step+t_granularity/2.) for t in range(ie1,ie2+1)]

            data_segments.append((key, fealength, actions, points, biases))

        if self.shuffle:
            random.shuffle(data_segments)

        return data_segments


    def gen_train_batch(self, index):

        batchdata = self.data_segments[index*self.batch:(index+1)*self.batch]
        aa, pp, bb, ff, mm = [], [], [], [], []

        for data in batchdata:
            a = np.zeros([1, self.args.out_window, self.args.class_num])
            p = np.zeros([1, 2, self.args.out_window, self.args.class_num])
            b = np.zeros([1, 2, self.args.out_window, self.args.class_num])
            f = np.zeros([1, self.args.in_window, 2048])
            m = np.zeros([1, self.args.out_window, 1])

            key, fealength, actions, points, biases = data

            features = np.load(os.path.join(self.feature, key+'.npy'))
            length = features.shape[0]

            if fealength <= self.args.out_window:
                a[0,:fealength,:] = actions
                p[0,:,:fealength,:] = points
                b[0,:,:fealength,:] = biases
                f[0,:length,:] = features
                m[0,:fealength,:] = 1
            else:
                actions_sum = np.sum(actions, 1)
                flag = 0
                count = 0
                while flag == 0:
                    count += 1
                    s = np.random.randint(0, fealength-self.args.out_window+1)
                    e = s + self.args.out_window
                    if (s == 0 or actions_sum[s] == 0) and (e == fealength or actions_sum[e-1] == 0):
                        a[0,:fealength,:] = actions[s:e,:]
                        p[0,:,:fealength,:] = points[:,s:e,:]
                        b[0,:,:fealength,:] = biases[:,s:e,:]
                        tmp_length = features[s*self.args.down_sample:e*self.args.down_sample,:].shape[0]
                        f[0,:tmp_length,:] = features[s*self.args.down_sample:e*self.args.down_sample,:]
                        m[0,:,:] = 1
                        flag = 1
                    if count > 1000:
                        break
                if flag == 0:
                    # print('no good sample')
                    a[0,:fealength,:] = actions[0:self.args.out_window,:]
                    p[0,:,:fealength,:] = points[:,0:self.args.out_window,:]
                    b[0,:,:fealength,:] = biases[:,0:self.args.out_window,:]
                    f[0,:,:] = features[0:self.args.in_window,:]
                    m[0,:,:] = 1
            aa.append(a)
            pp.append(p)
            bb.append(b)
            ff.append(f)
            mm.append(m)
        aa = np.concatenate(aa)
        pp = np.concatenate(pp)
        bb = np.concatenate(bb)
        ff = np.concatenate(ff)
        mm = np.concatenate(mm)

        return np.max(aa,2,keepdims=2), np.max(pp,3,keepdims=3), np.max(bb,3,keepdims=3)+np.min(bb,3,keepdims=3), ff, mm


    def gen_eval_batch(self, index):


        key, fealength, actions, points, biases = self.data_segments[index]

        features = np.load(os.path.join(self.feature, key+'.npy'))

        aa = np.expand_dims(actions, 0)
        pp = np.expand_dims(points, 0)
        bb = np.expand_dims(biases, 0)
        ff = np.expand_dims(features, 0)
        mm = np.ones((1,fealength,1))

        return key, np.max(aa,2,keepdims=2), np.max(pp,3,keepdims=3), np.max(bb,3,keepdims=3)+np.min(bb,3,keepdims=3), ff, mm




class pem_DataLoader():
    def __init__(self, batch, shuffle, datafile, evaluation=False):

        self.data = pickle.load(open(datafile, 'rb'))
        self.keys = list(self.data.keys())
        self.batch = batch
        self.num = len(self.keys)
        if shuffle:
            random.shuffle(self.keys)

        if evaluation:
            pass
        else:
            ratio = 0.9
            self.train_key = self.keys[:int(self.num*ratio)]
            self.val_key = self.keys[int(self.num*ratio):]

            self.train_data = []
            with tqdm(total=len(self.train_key)) as count:
                for key in self.train_key:
                    self.train_data += self.data[key]
                    count.update(1)
            self.train_num = len(self.train_data)
            self.train_nbatch = int(self.train_num / batch)

            self.val_data = []
            with tqdm(total=len(self.val_key)) as count:
                for key in self.val_key:
                    self.val_data += self.data[key]
                    count.update(1)
            self.val_num = len(self.val_data)
            self.val_nbatch = int(self.val_num / batch)

    def generate_batch(self, mode, step):

        if mode == 'train':
            pem_data = self.train_data
        else:
            pem_data = self.val_data
        feature = []
        iou = []
        for item in pem_data[step*self.batch:(step+1)*self.batch]:
            feature.append(item[0])
            iou.append(item[1])
        feature = np.vstack(feature)
        iou = np.vstack(iou)

        return feature, iou
















