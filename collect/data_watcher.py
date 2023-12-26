import pickle
datas = pickle.load(open('output/12_25/12_25_21_59_00.pkl', 'rb'))
print(len(datas))
for i in range(len(datas)):
    data = datas[i]
    print(data['robot_status']['jointAngle'], data['gripper_status'], data['gripper_pos'])
    open('tmp/wrist%s.jpg'%i, 'wb').write(data['wrist'])
    open('tmp/front%s.jpg'%i, 'wb').write(data['front'])
    open('tmp/right%s.jpg'%i, 'wb').write(data['right'])
    open('tmp/top%s.jpg'%i, 'wb').write(data['top'])