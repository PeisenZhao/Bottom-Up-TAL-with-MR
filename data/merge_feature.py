import numpy as np
import os
import tqdm

def merge_feature(mode):

    rgb_path = './THUMOS14/THUMOS14_FEATURES/I3D-4/thumos-{}-rgb-oversample'.format(mode)
    flow_path = './THUMOS14/THUMOS14_FEATURES/I3D-4/thumos-{}-flow-oversample'.format(mode)
    new_path = './I3D_features'

    if not os.path.exists(new_path):
        os.makedirs(new_path)

    rgb_list = os.listdir(rgb_path)
    flow_list = os.listdir(flow_path)

    # check flow and rgb
    rgb_names = [name.split('-')[0] for name in rgb_list]
    flow_names = [name.split('-')[0] for name in flow_list]

    assert len(rgb_names) == len(flow_names)
    rgb_names.sort()
    flow_names.sort()
    for i in range(len(rgb_names)):
        assert rgb_names[i] == flow_names[i]

    with tqdm.tqdm(len(rgb_names)) as count:

        for video in rgb_names:
            rgb_dict = dict(np.load(os.path.join(rgb_path, video+'-rgb.npz')))
            flow_dict = dict(np.load(os.path.join(flow_path, video+'-flow.npz')))

            rgb_fea = rgb_dict['feature'].mean(0)
            flow_fea = flow_dict['feature'].mean(0)

            concat_fea = np.concatenate([rgb_fea, flow_fea], 1)
            np.save(os.path.join(new_path, video+'.npy'), concat_fea)

            count.update(1)


if __name__ == '__main__':
    merge_feature('val')
    merge_feature('test')