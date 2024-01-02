import pickle
import os
import numpy as np
import random
import matplotlib.pyplot as plt

all_file = []

step = 1

for root, dirs, files in os.walk('.\collect\\output\\12_25', topdown=False):
    for name in files:
        if not name.endswith('.pkl'):
            continue
        filename = os.path.join(root, name)
        all_file.append(filename)
print(all_file)

epsoides = []
max_length = 32
all_show = []
min_p = np.array([-30, 0, 0, -30, 20, 0])
max_p = np.array([30, 100, 100, 10, 100, 1])
min_p = 900
max_p = 1350
for file in all_file:
    datas = pickle.load(open(file, 'rb'))
    print(len(datas))
    epsoide = {'name': file.split("/")[-1].split(".")[0],
               'angle_offset': [],
               'angle_pos': [],
               'gripper': [],
               'angle': [],
               'radians': [],
               'gripper_pos': [],
               'radians_pos': [],
               'image': {
                   'wrist': [],
                   'front': [],
                   'right': [],
                   'top': []
               }}
    if (len(datas)) < max_length:
        continue
    for j in range(max_length):
        i = j + 1
        data = datas[i]
        if i < len(datas) - step:
            next_data = datas[i + step]
        else:
            next_data = datas[-1]
        robot_status = data['robot_status']
        next_robot_status = next_data['robot_status']
        angle_offset = np.array(next_robot_status['jointAngle']) - np.array(robot_status['jointAngle'])
        gripper = data['gripper_status']
        # print(angle_offset, gripper, data['gripper_pos'])
        epsoide['angle_offset'].append(angle_offset)
        epsoide['angle'].append(np.array(next_robot_status['jointAngle']))
        epsoide['radians'].append(np.radians(next_robot_status['jointAngle']))
        epsoide['radians_pos'].append(np.radians(robot_status['jointAngle']))

        epsoide['gripper'].append([gripper])
        epsoide['gripper_pos'].append([(int(next_data['gripper_pos']) - min_p) / (max_p - min_p)])
        # print(np.radians(next_robot_status['jointAngle']), np.radians(robot_status['jointAngle']))
        all_show.append(np.radians(next_robot_status['jointAngle']))
        # jointAngle = (np.clip(np.array(robot_status['jointAngle']), min_p, max_p) - min_p) / (max_p - min_p)
        #
        # epsoide['angle_pos'].append(jointAngle)
        filename = file.split("\\")[-1].split(".")[0]
        epsoide['image']['wrist'].append('./image/%s_wrist_%s.jpg' % (filename, i))
        epsoide['image']['front'].append('./image/%s_front_%s.jpg' % (filename, i))
        epsoide['image']['top'].append('./image/%s_top_%s.jpg' % (filename, i))
        epsoide['image']['right'].append('./image/%s_right_%s.jpg' % (filename, i))

        if not os.path.exists('./image/%s_wrist_%s.jpg' % (filename, i)):
            open('./image/%s_wrist_%s.jpg' % (filename, i), 'wb').write(data['wrist'])
        if not os.path.exists('./image/%s_front_%s.jpg' % (filename, i)):
            open('./image/%s_front_%s.jpg' % (filename, i), 'wb').write(data['front'])
        if not os.path.exists('./image/%s_top_%s.jpg' % (filename, i)):
            open('./image/%s_top_%s.jpg' % (filename, i), 'wb').write(data['top'])
        if not os.path.exists('./image/%s_right_%s.jpg' % (filename, i)):
            open('./image/%s_right_%s.jpg' % (filename, i), 'wb').write(data['right'])
    # print(epsoide)
    epsoides.append(epsoide)
    # break

print(np.array(all_show)[:, 1].shape)
plt.figure(1)
plt.hist(np.array(all_show)[:, 1], bins=256, facecolor="blue", edgecolor="black", alpha=0.7)
plt.xlabel("x")
plt.ylabel("dis")
plt.show()
# -30, 30
# 0, 100
# 0, 100
# -25, 10
# 20, 100
# 0
random.shuffle(epsoides)
train_ratio = 0.9
train = epsoides[:int(len(epsoides)*train_ratio)]
test = epsoides[int(len(epsoides)*train_ratio):]
pickle.dump(train, open('train_data.pkl', 'wb'))
pickle.dump(test, open('test.pkl', 'wb'))
print(len(train), len(test))