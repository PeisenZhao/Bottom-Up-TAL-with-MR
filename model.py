import tensorflow as tf

import ipdb
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.models import BatchNorm


class MultiDenseNet():
    def __init__(self, args, feature, action, point, bias, mask):

        self.args = args
        self.model_input = feature # [N,window,C]
        self.gt_action = action # [N,window,class]
        self.gt_point = point # [N,2,window,class]
        self.gt_bias = bias # [N,2,window,class]
        self.mask = mask # [N,window,1]
        self.loss = {}
        self.BuildModel()
        self.lossfunc()
        self.lr = tf.placeholder(tf.float32, [None], 'lr')
        self.solver = tf.train.MomentumOptimizer(learning_rate=self.lr[0], momentum=0.9).minimize(self.loss['loss'])

    def BuildModel(self):

        baselayer1 = tf.layers.conv1d(inputs=self.model_input,filters=512,kernel_size=9,dilation_rate=1,strides=2,padding='same',activation=tf.nn.relu,name='baselayer1')
        baselayer2 = tf.layers.conv1d(inputs=baselayer1,filters=512,kernel_size=9,dilation_rate=1,strides=2,padding='same',activation=tf.nn.relu,name='baselayer2')
        baselayer = baselayer2
        self.action_heat_fea = tf.layers.conv1d(inputs=baselayer,filters=256,kernel_size=5,dilation_rate=1,strides=1,padding='same',activation=tf.nn.relu,name='heat_action1')
        self.action_heat = tf.layers.conv1d(inputs=self.action_heat_fea,filters=1,kernel_size=5,dilation_rate=1,strides=1,padding='same',activation=tf.nn.sigmoid,name='heat_action2')
        self.start_heat_fea = tf.layers.conv1d(inputs=baselayer,filters=256,kernel_size=5,dilation_rate=1,strides=1,padding='same',activation=tf.nn.relu,name='heat_start1')
        self.start_heat = tf.layers.conv1d(inputs=self.start_heat_fea,filters=1,kernel_size=5,dilation_rate=1,strides=1,padding='same',activation=tf.nn.sigmoid,name='heat_start2')
        self.end_heat_fea = tf.layers.conv1d(inputs=baselayer,filters=256,kernel_size=5,dilation_rate=1,strides=1,padding='same',activation=tf.nn.relu,name='heat_end1')
        self.end_heat = tf.layers.conv1d(inputs=self.end_heat_fea,filters=1,kernel_size=5,dilation_rate=1,strides=1,padding='same',activation=tf.nn.sigmoid,name='heat_end2')
        start_bias = tf.layers.conv1d(inputs=baselayer,filters=256,kernel_size=5,dilation_rate=1,strides=1,padding='same',activation=tf.nn.relu,name='start_bias1')
        self.start_bias = tf.layers.conv1d(inputs=start_bias,filters=1,kernel_size=5,dilation_rate=1,strides=1,padding='same',activation=tf.identity,name='start_bias2')
        end_bias = tf.layers.conv1d(inputs=baselayer,filters=256,kernel_size=5,dilation_rate=1,strides=1,padding='same',activation=tf.nn.relu,name='end_bias1')
        self.end_bias = tf.layers.conv1d(inputs=end_bias,filters=1,kernel_size=5,dilation_rate=1,strides=1,padding='same',activation=tf.identity,name='end_bias2')


    def lossfunc(self):

        def _regr_loss(regr, gt, weight, mode):

            regr_loss = tf.losses.huber_loss(gt, regr, weight)
            self.loss['{}_regrloss'.format(mode)] = regr_loss

            return regr_loss

        def _heat_loss(heat, gt, mode):

            # gt, heat -> [N,window,class]

            pmask = tf.cast(gt, dtype=tf.float32)  # [N,window,class]
            nmask = self.mask - pmask  # [N,window,class]
            pos_num = tf.reduce_sum(pmask, 1) # [N,class]
            neg_num = tf.reduce_sum(nmask, 1) # [N,class]
            pos_loss = - tf.reduce_mean(tf.reduce_sum(pmask*tf.log(heat+1e-7), 1)/(pos_num+1e-7))
            neg_loss = - tf.reduce_mean(tf.reduce_sum(nmask*tf.log(1.0-heat+1e-7), 1)/(neg_num+1e-7))
            self.loss['{}_heatloss_pos'.format(mode)] = pos_loss
            self.loss['{}_heatloss_neg'.format(mode)] = neg_loss
            ce_loss = 0.5 * (pos_loss + neg_loss)
            self.loss['{}_heatloss'.format(mode)] = ce_loss

            return ce_loss


        def _intra_consistency_loss(heat, gt, mode):

            # heat_fea -> [N,window,1]
            # gt -> [N,window,class]
            # mask -> [N,window,1]

            # using cosine distance

            # a = np.array([[0,1,1,1,1,0,0,0]]) shape: (1,8)
            # matmul(a.T, a)
            # Out: 
            #        [0, 0, 0, 0, 0, 0, 0, 0]             [0, 0, 0, 0, 0, 0, 0, 0]            [0, 1, 1, 1, 1, 0, 0, 0]           [0, 0, 0, 0, 0, 1, 1, 1]
            #        [0, 1, 1, 1, 1, 0, 0, 0]             [0, 0, 1, 1, 1, 0, 0, 0]            [1, 0, 0, 0, 0, 1, 1, 1]           [0, 0, 0, 0, 0, 0, 0, 0]
            #        [0, 1, 1, 1, 1, 0, 0, 0]             [0, 1, 0, 1, 1, 0, 0, 0]            [1, 0, 0, 0, 0, 1, 1, 1]           [0, 0, 0, 0, 0, 0, 0, 0]
            #        [0, 1, 1, 1, 1, 0, 0, 0]    ---->    [0, 1, 1, 0, 1, 0, 0, 0]            [1, 0, 0, 0, 0, 1, 1, 1]           [0, 0, 0, 0, 0, 0, 0, 0]
            #        [0, 1, 1, 1, 1, 0, 0, 0]             [0, 1, 1, 1, 0, 0, 0, 0]      &     [1, 0, 0, 0, 0, 1, 1, 1]           [0, 0, 0, 0, 0, 0, 0, 0]
            #        [0, 0, 0, 0, 0, 0, 0, 0]             [0, 0, 0, 0, 0, 0, 0, 0]            [0, 1, 1, 1, 1, 0, 0, 0]           [1, 0, 0, 0, 0, 0, 1, 1]
            #        [0, 0, 0, 0, 0, 0, 0, 0]             [0, 0, 0, 0, 0, 0, 0, 0]            [0, 1, 1, 1, 1, 0, 0, 0]           [1, 0, 0, 0, 0, 1, 0, 1]
            #        [0, 0, 0, 0, 0, 0, 0, 0]             [0, 0, 0, 0, 0, 0, 0, 0] M_gt_1     [0, 1, 1, 1, 1, 0, 0, 0] M_gt_2    [1, 0, 0, 0, 0, 1, 1, 0] M_gt_3  


            distance = tf.abs(heat - tf.transpose(heat, [0,2,1])) # [N,window,window]

            # gt -> [N,window,class]

            mask = self.mask
            mask = tf.matmul(mask, tf.transpose(mask, [0,2,1])) # [N,window,window]
            M_gt_1 = tf.nn.relu(tf.matmul(gt, tf.transpose(gt, [0,2,1])) - tf.eye(self.args.out_window, batch_shape=[self.args.batch])) # [N,window,window]
            M_gt_2 = tf.abs(gt - tf.transpose(gt, [0,2,1])) # [N,window,window]
            M_gt_3 = tf.ones_like(M_gt_1) - tf.eye(self.args.out_window, batch_shape=[self.args.batch]) - M_gt_1 - M_gt_2

            M_gt_1 = M_gt_1 * mask # [N,window,window]
            M_gt_2 = M_gt_2 * mask # [N,window,window]
            M_gt_3 = M_gt_3 * mask # [N,window,window]

            pairs_1 = tf.reduce_sum(M_gt_1, [1,2]) + 1e-7 # [N]
            pairs_2 = tf.reduce_sum(M_gt_2, [1,2]) + 1e-7 # [N]
            pairs_3 = tf.reduce_sum(M_gt_3, [1,2]) + 1e-7 # [N]

            consistency_1 = tf.reduce_mean(tf.reduce_sum(distance * M_gt_1, [1,2]) / pairs_1) # [1]
            consistency_2 = 1 - tf.reduce_mean(tf.reduce_sum(distance * M_gt_2, [1,2]) / pairs_2) # [1]
            consistency_3 = tf.reduce_mean(tf.reduce_sum(distance * M_gt_3, [1,2]) / pairs_3) # [1]

            consistency_loss = consistency_1 + consistency_2 + consistency_3

            self.loss['intra_consistency_{}_1'.format(mode)] = consistency_1
            self.loss['intra_consistency_{}_2'.format(mode)] = consistency_2
            self.loss['intra_consistency_{}_3'.format(mode)] = consistency_3
            self.loss['intra_consistency_{}'.format(mode)] = consistency_loss

            return consistency_loss


        def _inter_consistency_loss(action_heat, start_heat, end_heat):

            # action_heat -> [N,window,1]
            # start_heat -> [N,window,1]
            # end_heat -> [N,window,1]
            # mask -> [N,window,1]

            diff = tf.concat([action_heat[:,1:,]-action_heat[:,:-1,], action_heat[:,-1:,]-action_heat[:,-2:-1,]], 1) # [N,window,1]
            # diff = diff / tf.reduce_max(diff, [1,2]) # [N,window,1]
            diff_1 = tf.where(tf.greater_equal(diff, 0), diff, tf.zeros_like(diff))
            diff_0 = -tf.where(tf.less_equal(diff, 0), diff, tf.zeros_like(diff))
            # start_heat = start_heat / tf.reduce_max(start_heat, [1,2])
            # end_heat = end_heat / tf.reduce_max(end_heat, [1,2])

            start_diff_consistency = tf.reduce_mean(tf.abs(diff_1 - start_heat))
            end_diff_consistency = tf.reduce_mean(tf.abs(diff_0 - end_heat))            
            consistency_loss = end_diff_consistency + start_diff_consistency
            self.loss['inter_consistency_start'] = start_diff_consistency
            self.loss['inter_consistency_end'] = end_diff_consistency

            return consistency_loss


        # heat loss
        action_heatloss = _heat_loss(self.action_heat, self.gt_action, 'action')
        start_heatloss = _heat_loss(self.start_heat, self.gt_point[:,0,:,:], 'start')
        end_heatloss = _heat_loss(self.end_heat, self.gt_point[:,1,:,:], 'end')

        # regr loss
        start_regrloss = _regr_loss(self.start_bias, self.gt_bias[:,0], tf.cast(self.gt_point[:,0,:,:], dtype=tf.float32), 'start')
        end_regrloss = _regr_loss(self.end_bias, self.gt_bias[:,1], tf.cast(self.gt_point[:,1,:,:], dtype=tf.float32), 'end')

        # consistency_loss
        action_consistency = _intra_consistency_loss(self.action_heat, self.gt_action, 'action')
        start_consistency = _intra_consistency_loss(self.start_heat, self.gt_point[:,0,:,:], 'start')
        end_consistency = _intra_consistency_loss(self.end_heat, self.gt_point[:,1,:,:], 'end')
        inter_consistency = _inter_consistency_loss(self.action_heat, self.start_heat, self.end_heat)

        self.loss['heatloss'] = action_heatloss + start_heatloss + end_heatloss
        self.loss['regrloss'] = start_regrloss + end_regrloss
        self.loss['intra_consistency_loss'] = start_consistency + end_consistency + action_consistency
        self.loss['inter_consistency_loss'] = inter_consistency
        self.loss['loss'] = self.loss['heatloss'] + self.loss['regrloss'] + self.loss['intra_consistency_loss'] + self.loss['inter_consistency_loss']



class pemNet():
    def __init__(self, feature, iou):
        self.feature = feature
        self.iou = iou
        self.loss = {}

        self.BulidModel()
        self.lossfunc()

        self.lr = tf.placeholder(tf.float32, [None], 'lr')
        self.solver = tf.train.MomentumOptimizer(learning_rate=self.lr[0], momentum=0.9).minimize(self.loss['loss'])

    def BulidModel(self):

        layer1 = tf.layers.dense(inputs=self.feature,units=96,activation=tf.nn.relu)
        layer2 = tf.layers.dense(inputs=layer1,units=48,activation=tf.nn.relu)
        self.output = tf.layers.dense(inputs=layer2,units=1,activation=tf.nn.sigmoid)

    def lossfunc(self):

        self.loss['loss'] = tf.losses.mean_squared_error(self.iou, self.output)


