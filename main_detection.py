
import json
import os
import numpy as np


from eval import evaluation_detection
from configuration import Logger, parse_base_args
from collections import Counter


label_id = [7, 9, 12, 21, 22, 23, 24, 26, 31, 33, 36, 40, 45, 51, 68, 79, 85, 92, 93, 97]
label_id = [i-1 for i in label_id]
label_name = ['BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk', 'CliffDiving', 
              'CricketBowling', 'CricketShot', 'Diving', 'FrisbeeCatch', 'GolfSwing', 'HammerThrow', 
              'HighJump', 'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput', 'SoccerPenalty', 'TennisSwing', 
              'ThrowDiscus', 'VolleyballSpiking']
 

def iou_score(gt, anchor):
    gt_min, gt_max = gt
    an_min, an_max = anchor
    if (an_min >= gt_max) or (gt_min >= an_max):
        return 0.
    else:
        union = max(gt_max, an_max) - min(gt_min, an_min)
        inter = min(gt_max, an_max) - max(gt_min, an_min)
        return float(inter) / union


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def AddPreLabel(original, output):


    un = np.load('./Evaluation/data/uNet_test.npy')
    un_new = un[:,label_id]

    data = json.load(open(original))
    with open('./Evaluation/data/detclasslist.txt', 'r') as f:
        lines = f.readlines()

    out =  {'version':'THUMOS14', 'results':{}, 'external_data':'{}'}
    out['results'] = {vid:[] for vid in data['results'].keys()}

    for videoid, v in data['results'].items():
        print(videoid)
        vid = int(videoid.split('_')[-1])

        tmp_list = []

        for result in v:
            topK = 1
            class_tops = np.argsort(un_new[vid-1])[::-1]
            class_scores = np.sort(softmax(un_new[vid-1]))[::-1]

            for i in range(topK):

                score = result['score']#* class_scores[i]
                label = label_name[class_tops[i]]
                tmp_list.append({'score':score,'label':label,'segment':result['segment']})

        index = np.argsort([t['score'] for t in tmp_list])
        tmp_list = [tmp_list[idx] for idx in index[::-1]]

        if len(tmp_list)>200:
            out['results'][videoid] = tmp_list[:200]
        else:
            out['results'][videoid] = tmp_list


    with open(output, 'w') as f:
        json.dump(out, f)


if __name__ == '__main__':

    args = parse_base_args()
    save_path = args.save_path

    for epoch in range(1, args.epochs+1):

        in_json = os.path.join(save_path, 'proposals', 'results_softnms_n5_score_se_pem_epoch{}.json'.format(epoch))
        out_json = os.path.join(save_path, 'proposals', 'results_softnms_detection_se_pem_epoch{}.json'.format(epoch))
        AddPreLabel(in_json, out_json)

    mAP = []

    for epoch in range(1, args.epochs+1):
        eval_file = os.path.join(save_path, 'proposals', 'results_softnms_detection_se_pem_epoch{}.json'.format(epoch))
        results = evaluation_detection(args, eval_file)
        mAP.append(results)
    mAP = np.array(mAP)
    mean_10 = np.mean(mAP[10:,:],0)

    with open(os.path.join(save_path, 'result_softnms_n5_detection_map.txt'), 'w') as f:
        for i in range(20):
            s = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(i+1,mAP[i][0],mAP[i][1],mAP[i][2],mAP[i][3],mAP[i][4],mAP[i][5],mAP[i][6],mAP[i][7],mAP[i][8])
            f.write(s)
        s = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format('averaged_last10',mean_10[0],mean_10[1],mean_10[2],mean_10[3],mean_10[4],mean_10[5],mean_10[6],mean_10[7],mean_10[8])
        f.write(s)


