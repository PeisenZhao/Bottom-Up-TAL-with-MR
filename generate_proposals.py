import os
import pickle
import numpy as np
import json
from scipy.interpolate import interp1d
from model import pemNet



def iou_score(gt, anchor):
    gt_min, gt_max = gt
    an_min, an_max = anchor
    if (an_min >= gt_max) or (gt_min >= an_max):
        return 0.
    else:
        union = max(gt_max, an_max) - min(gt_min, an_min)
        inter = min(gt_max, an_max) - max(gt_min, an_min)
        return float(inter) / union


def nms(pairs):

    # pairs is a list : [[s,e,score], [s,e,score], ...]

    score = [d[2] for d in pairs]
    index = np.argsort(score)[::-1]

    pairs = [pairs[i] for i in index]
    filter_pairs = []

    while len(pairs) > 0 and len(filter_pairs) < 1001:
        if len(pairs) == 1:
            candidate = pairs[0]
            filter_pairs.append(candidate)
            break
        else:
            candidate = pairs[0]
            filter_pairs.append(candidate)
            others = pairs[1:]
            pairs = [pair for pair in others if iou_score(pair[:2],candidate[:2])<0.7]

    return filter_pairs



def soft_nms(pairs):

    # pairs is a list : [[s,e,score], [s,e,score], ...]

    score = [d[2] for d in pairs]
    index = np.argsort(score)[::-1]
    pairs = [pairs[i] for i in index]

    filter_pairs = []

    while len(pairs) > 0 and len(filter_pairs) < 1001:
        if len(pairs) == 1:
            candidate = pairs[0]
            filter_pairs.append(candidate)
            break
        else:
            candidate = pairs[0]
            filter_pairs.append(candidate)
            others = pairs[1:]
            new_pairs = []
            for pair in others:
                iou = iou_score(pair[:2],candidate[:2])
                if iou > 0.7:
                    new_pairs.append(pair[:2]+[pair[2]*np.exp(-np.square(iou)/0.75)])
                else:
                    new_pairs.append(pair)

            # pairs = [pair[:2]+[pair[2]*np.exp(-np.square(iou_score(pair[:2],candidate[:2]))/0.75)] for pair in others if iou_score(pair[:2],candidate[:2])>0.7 else pair]

            score = [d[2] for d in new_pairs]
            index = np.argsort(score)[::-1]
            pairs = [new_pairs[i] for i in index]

    return filter_pairs



def generate_pem_data(args, pairs, key, predictheat):

    # pairs ([[1.1s, 2.3s, 0.94], [...], ...])
    # key video name

    data_path = args.data_path
    anno_path = os.path.join(data_path, 'annotation')
    alldata = json.load(open(os.path.join(anno_path, 'thumos14.json')))['database']

    t_step = args.t_step / args.fps[key]
    t_granularity = args.t_granularity / args.fps[key]
    length = int((alldata[key]['fealength_step4']+args.down_sample-1) / args.down_sample)
    t_length = t_granularity/2. + t_step*(length-1)

    granularity_list = [t_granularity/2. + t_step*(l) for l in range(length)]

    most_small = granularity_list[0]+1e-2
    most_large = granularity_list[-1]-1e-2

    af = interp1d(granularity_list, predictheat['action_heat'][:length])
    sf = interp1d(granularity_list, predictheat['start_heat'][:length])
    ef = interp1d(granularity_list, predictheat['end_heat'][:length])

    pem_data = []

    for pair in pairs:
        ps = pair[0]
        pe = pair[1]
        duration = pe - ps
        iou_list = []
        for gt in alldata[key]['annotations']:
            iou_list.append(iou_score([ps,pe], gt['segment']))
        iou = max(iou_list)

        ps_new = min(max(ps-0.2*duration, most_small), most_large)
        pe_new = max(min(most_large, pe+0.2*duration), most_small)
        step = (pe_new - ps_new)/31
        index = [ps_new + step*i for i in range(32)]
        # print(index[-1], granularity_list[-1], '---', index[0], granularity_list[0])
        pem_fea = np.hstack([sf(index),af(index),ef(index)])
        pem_data.append([pem_fea,iou,pair[2]])

    return pem_data


def generate_proposals(args, epoch, mode, prepare_pemdata=False):

    picklefile = os.path.join(args.save_path, 'predicts', '{}_results_epoch{}.pickle'.format(mode, epoch))
    results = pickle.load(open(picklefile, 'rb'))
    proposal_results = {'version':'THUMOS14', 'results':{}, 'external_data':{}}
    pem_dict = {}
    for key, data in results.items():

        # print(key, data['action_heat'].shape)
        action_heat = data['action_heat']
        start_heat = data['start_heat']
        end_heat = data['end_heat']
        start_regr = data['start_regr']
        end_regr = data['end_regr']
        start_thred = 0.5 * (start_heat.max() + start_heat.min())
        end_thred = 0.5 * (end_heat.max() + end_heat.min())

        # print start_thred, end_thred
        starts = np.array(start_heat > start_thred, dtype=np.int32)
        ends = np.array(end_heat > end_thred, dtype=np.int32)

        # add peak points
        for i in range(1,len(start_heat)-1):
            if start_heat[i] > start_heat[i-1] and start_heat[i] > start_heat[i+1]:
                if starts[i] == 0:
                    starts[i] += 1
        for i in range(1,len(end_heat)-1):
            if end_heat[i] > end_heat[i-1] and end_heat[i] > end_heat[i+1]:
                if ends[i] == 0:
                    ends[i] += 1

        # no start no end situation
        if np.sum(starts) == 0:
            starts[0] = 1
            print('no predicted start')
        if np.sum(ends) == 0:
            ends[-1] = 1
            print('no predicted end')

        # prepare start end list 
        start_list = []
        end_list = []
        for i in range(len(starts)):
            if starts[i] == 1:
                start_list.append(i)
            if ends[i] == 1:
                end_list.append(i)
        if min(start_list)>=max(end_list):
            start_list.append(0)
            end_list.append(len(ends)-1)
            print('flag', key)

        # prepare pairs with embedding distance
        pairs = []
        for i in range(len(start_list)):
            count = 0
            for j in range(len(end_list)):
                if end_list[j] <= start_list[i]:
                    continue
                if end_list[j] - start_list[i] > 190:
                    continue
                if end_list[j] > start_list[i]:
                    count += 1
                if count > 5:
                    continue
                s = start_list[i]
                e = end_list[j]
                actionness = np.mean(action_heat[s:e+1])
                startness = start_heat[s]
                endness = end_heat[e]
                pairs.append([s,e,float(startness*endness)])
                # pairs.append([s,e,(actionness+startness+endness)/3.])

        t_step = args.t_step / args.fps[key]
        t_granularity = args.t_granularity / args.fps[key]
        pairs_add_bias = [[t_granularity/2+t_step*pair[0]+start_regr[pair[0]], t_granularity/2+t_step*pair[1]+end_regr[pair[1]], pair[2]] for pair in pairs]
        pairs_after_nms = soft_nms(pairs_add_bias)
        segments = []
        for pair in pairs_after_nms:
            segment = {}
            # segment['score'] = np.random.random()
            segment['score'] = pair[2]  
            segment['segment'] = [pair[0], pair[1]]
            segments.append(segment)
        proposal_results['results'][key] = segments
        if prepare_pemdata:
            pem_data = generate_pem_data(args,pairs_after_nms,key,data)
            pem_dict[key] = pem_data

    if not os.path.exists(os.path.join(args.save_path, 'proposals')):
        os.makedirs(os.path.join(args.save_path, 'proposals'))

    if prepare_pemdata:
        with open(os.path.join(args.save_path, 'proposals', 'pem_data_n5_epoch{}.pickle'.format(epoch)), 'wb') as f:
            pickle.dump(pem_dict, f)
    else:
        with open(os.path.join(args.save_path, 'proposals', 'results_softnms_n5_score_se_epoch{}.json'.format(epoch)), 'w') as f:
            json.dump(proposal_results, f)




