import sys, os
sys.path.append('./Evaluation')
from eval_proposal import ANETproposal
from eval_detection import ANETdetection
import numpy as np



def evaluation_proposal(args, eval_file):

    ground_truth_filename = './Evaluation/data/thumos14.json'
    anet_proposal = ANETproposal(ground_truth_filename, eval_file,
                                 tiou_thresholds=np.linspace(0.5, 1.0, 11),
                                 max_avg_nr_proposals=1000,
                                 subset='test', verbose=True, check_status=False)
    anet_proposal.evaluate()
    recall = anet_proposal.recall
    average_recall = anet_proposal.avg_recall
    average_nr_proposals = anet_proposal.proposals_per_video
    names = ['AR@50', 'AR@100', 'AR@200', 'AR@500', 'AR@1000']
    values = [np.mean(recall[:,i]) for i in [49, 99, 199, 499, 999]]
    return names, values



def evaluation_detection(args, eval_file):

    ground_truth_filename = './Evaluation/data/thumos14.json'
    anet_detection = ANETdetection(ground_truth_filename, eval_file,
                                 tiou_thresholds=np.linspace(0.1, 0.9, 9),
                                 subset='test', verbose=True, check_status=False)
    anet_detection.evaluate()
    ap = anet_detection.ap
    mAP = anet_detection.mAP
    return mAP
