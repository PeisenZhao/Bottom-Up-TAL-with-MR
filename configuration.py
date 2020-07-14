import argparse
import os
import sys

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

def build_info(args):

    # training
    args.batch = 3
    args.epochs = 20
    args.pem_epochs = 40
    args.pem_batch = 256

    # dataset
    args.data_path = './data'
    args.in_window = 3000
    args.out_window = 750
    args.t_granularity = 16 # 16frame/25frame  fps:25frames
    args.t_step = 16 # 4frame/25frame 
    args.down_sample = 4
    args.class_num = 20

    with open(os.path.join(args.data_path, 'videos_fps.txt'), 'r') as f:
        lines = f.readlines()
    args.fps = {}
    for line in lines:
        tmp = line.strip().split()
        args.fps[tmp[0]] = float(tmp[1])

    # save info
    save_info = [('batch', args.batch), ('window', args.out_window),
                 ('epochs', args.epochs), ('model', 'kernel_9_9_5_5')
                ]
    save_name = ''
    for item in save_info:
        save_name += str(item[0])+'-'+str(item[1])+'|'
    args.save_path = './train_logs/{}'.format(save_name[:-1])
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    return args



def parse_base_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', type=str)

    return build_info(parser.parse_args())