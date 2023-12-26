import h5py
import numpy as np
from utils import load_data
import glob
from PIL import Image
# print(len(glob.glob("./data/*")))
# exit()

path = "data/making_brownies_place_bowl_scene_2_20230324-115210.h5"
h5 = h5py.File(path, 'r')
print(h5.keys()) #Outputs Trials per h5, Trial 0, 1, 2, ...
for key in h5['Trial19']['data'].keys(): # outputs data, derived, and config
    print(h5['Trial19']['data'][key])
print(np.array(h5['Trial19']['data']['rgb_left']))
for i in range(np.array(h5['Trial19']['data']['rgb_left']).shape[0]):
    Image.fromarray(np.array(h5['Trial19']['data']['rgb_left'])[i]).save('tmp/left%s.jpg' % i)
    Image.fromarray(np.array(h5['Trial19']['data']['rgb_right'])[i]).save('tmp/right%s.jpg' % i)
    Image.fromarray(np.array(h5['Trial19']['data']['rgb_top'])[i]).save('tmp/top%s.jpg' % i)
    Image.fromarray(np.array(h5['Trial19']['data']['rgb_wrist'])[i]).save('tmp/wrist%s.jpg' % i)

print('***')
print(np.array(h5['Trial19']['data']['ctrl_arm'])[:,0])
    # print(h5['Trial0']['data'][key])
print('=========')
# for key, trial in h5.items():
#     print(key, trial['data']['ctrl_arm'])
#     action = np.concatenate([trial['data']['ctrl_arm'], trial['data']['ctrl_ee']], axis=1).astype(np.float32)
#     print(np.array(trial['data']['qp_arm']))
#     break
# # # to extract the data
# h5["Trial0"]['data'][data_key] #where data_key is one of the cells from the data tab

# train_dataloader, val_dataloader, stats, is_sim = load_data('./data', 6000, 1, 1)
# print(train_dataloader[0])
