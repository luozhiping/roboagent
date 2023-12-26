from STR400_SDK.str400 import STR400
import time
import serial
import cv2
from threading import Thread
import pickle
from _datetime import datetime
import os
from policy import ACTPolicy, CNNMLPPolicy
import torch
import numpy as np
from constants import *
from PIL import Image
from robot_control_serial import RobotModbus

# 1.舵机位置#000PRAD!
# 2.sleep 1s 后开始记录
CAMERA_NAMES = ['wrist', 'front', 'right', 'top']

class Control:

    BEGIN_ANGLE = [-10, 0, 90, 0, 50, 0]
    END_ANGLE = [0, 0, 90, 0, 90, 0]
    GRIPPER_OPEN = 0
    GRIPPER_CLOSE = 1
    GRIPPER_OPENING = 2
    GRIPPER_CLOSING = 3
    def __init__(self):
        self.robot = STR400(host='localhost', port=8080)
        # self.robot = RobotModbus()
        # self.robot.open_system()
        self.ser = serial.Serial("COM9", 115200)  # 打开COM17，将波特率配置为115200，其余参数使用默认值
        # self.folder_name = "%s" % (datetime.now().strftime("%m_%d"))
        # os.makedirs("output/%s" % self.folder_name, exist_ok=True)
        np.set_printoptions(suppress=True)

        self.pos_start = None
        self.pos_end = None
        self.robot_status = None
        self.recording = False
        self.data = []
        self.gripper_status = Control.GRIPPER_CLOSE
        self.gripper_pos = 0
        # t = Thread(target=self.get_status)
        # t.start()

        # self.close_gripper()
        self.open_camera()
        self.open_gripper()
        self.begin_ai()
        # self.begin_mark()


    def load_model(self):
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        state_dim = 7
        lr_backbone = 1e-5
        backbone = 'resnet18'
        policy_config = {'lr': 1e-05,
                         'num_queries': 20,
                         'kl_weight': 10,
                         'hidden_dim': 512,
                         'dim_feedforward': 3200,
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         }
        config = {
            'num_epochs': 10,
            'ckpt_dir': './',
            'state_dim': state_dim,
            'lr': 1e-05,
            'real_robot': 'TBD',
            'policy_class': 'ACT',
            'policy_config': policy_config,
            'task_name': 'a',
            'seed': 0,
            'temporal_agg': True
        }

        policy_config['camera_names'] = CAMERA_NAMES
        config['camera_names'] = CAMERA_NAMES
        config['real_robot'] = True
        config['episode_len'] = 100
        ckpt_dir = '../'

        ckpt_names = [f'train1130/policy_best.ckpt']
        ckpt_name = ckpt_names[0]
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        torch.set_printoptions(precision=4, sci_mode=False)
        policy = ACTPolicy(policy_config)
        loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')))

        print(loading_status)
        if torch.cuda.is_available():
            policy.cuda()
        policy.eval()
        self.policy = policy
        print(f'Loaded: {ckpt_path}')
        stats_path = os.path.join(ckpt_dir, f'train1130/dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)

        self.post_process = lambda a: a * stats['action_std'] + stats['action_mean']
        self.norm_stats = stats
        # all_actions = policy(robot_status.unsqueeze(0), image_data[t].unsqueeze(0), task_emb=task_emb)

    def begin_ai(self):
        self.load_model()
        # self.robot.stop()
        # self.robot.disable()
        #
        self.robot.enable()
        self.move_to(Control.BEGIN_ANGLE + [3])
        # self.robot.move_to_angles(Control.BEGIN_ANGLE)

        for i in range(5):
            ret, image = self.cap.read()
            ret, image2 = self.cap2.read()
            ret, image3 = self.cap3.read()
            ret, image4 = self.cap4.read()
            cv2.imwrite('run_0.jpg', image)
            cv2.imwrite('run_1.jpg', image2)
            cv2.imwrite('run_2.jpg', image3)
            cv2.imwrite('run_3.jpg', image4)
            time.sleep(0.1)


        i = 0
        num_queries = 20
        all_time_actions = torch.zeros([1000, 1000 + num_queries, 7]).cuda()
        t = 0

        save_mode = False
        grasped = False

        while True:
            i += 1
            begin = time.time()
            ret, image = self.cap.read()
            ret, image2 = self.cap2.read()
            ret, image3 = self.cap3.read()
            ret, image4 = self.cap4.read()
            cv2.imwrite('./run/run_0_%s.jpg' % i, image)
            cv2.imwrite('./run/run_1_%s.jpg' % i, image2)
            cv2.imwrite('./run/run_2_%s.jpg' % i, image3)
            cv2.imwrite('./run/run_3_%s.jpg' % i, image4)

            print(np.array(image).shape)
            all_cam_images = [np.array(Image.open('./run/run_0_%s.jpg' % i)), np.array(Image.open('./run/run_1_%s.jpg' % i)),
                              np.array(Image.open('./run/run_2_%s.jpg' % i)), np.array(Image.open('./run/run_3_%s.jpg' % i))]
            all_cam_images = np.stack(all_cam_images, axis=0)
            image_data = torch.from_numpy(all_cam_images)
            image_data = torch.einsum('k h w c -> k c h w', image_data)
            image_data = image_data / 255.0
            min_p = np.array([-20, 0, 0, -30, 50, -50])
            max_p = np.array([10, 100, 100, 5, 110, 100])
            robot_status = self.robot.get_robot_status()
            jointAngle = (np.clip(np.array(robot_status['jointAngle']), min_p, max_p) - min_p) / (max_p - min_p)
            qpos = np.array(jointAngle).astype(np.float32)
            qpos_data = torch.from_numpy(qpos).float()
            qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

            task_emb = TEXT_EMBEDDINGS[0]
            task_emb = torch.from_numpy(np.asarray(task_emb)).float()
            with torch.inference_mode():
                all_actions = self.policy(qpos_data.unsqueeze(0).cuda(), image_data.unsqueeze(0).cuda(), task_emb=task_emb.unsqueeze(0).cuda())


                if True:
                    all_time_actions[[t], t:t + num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    raw_action = raw_action.squeeze(0).cpu().numpy()
                    action = self.post_process(raw_action)
                    action = action + np.random.normal(loc=0.0, scale=0.05, size=len(action))
                    now = self.post_process(all_actions[0][0].cpu())
                    print('当前指令:', now)
                    print('平均位移:', action)
                    if save_mode:
                        if action[1] > 0:
                            action[1] = -1
                            print('修改位移:', action)
                    target = np.array(robot_status['jointAngle']) + action[:6]
                    print('目标位置:', target)
                    if save_mode and not grasped:
                        # conti = input('继续吗?')
                        # if conti == 'y':
                        if now[6] > 2 and not grasped:
                            all_time_actions = torch.zeros([1000, 1000 + num_queries, 7]).cuda()
                            print('历史清零')
                            grasped = True
                            self.close_gripper()
                        else:
                            pass
                        # else:
                        #     break
                    move_status = self.move_abs(action[:6])
                    robot_status = self.robot.get_robot_status()
                    print('误差：', np.abs(np.array(robot_status['jointAngle']) - target))
                    if not grasped and move_status == 1:
                        print('到达位置，进入手动操作模式')
                        save_mode = True
                        continue
                    if save_mode and move_status != 1:
                        save_mode = False
                    if now[6] > 6 and grasped:
                        self.open_gripper()
                        print('任务完成，退出')
                        break


                else:
                    action = self.post_process(all_actions[0][0].cpu()).numpy()
                    robot_status = self.robot.get_robot_status()
                    print('infer:', action)
                    print('goto :', np.array(robot_status['jointAngle']) + action[:6])
                    # self.move_to(list(np.around(np.array(robot_status['jointAngle']) + action[:6], 5)) + [2])
                    self.move_abs(action[:6])
                    # time.sleep(2)
                t += 1

                robot_status = self.robot.get_robot_status()
                print('res  :', robot_status['jointAngle'])
            # break


    def get_images(self):
        ret, image = self.cap.read()
        ret, image2 = self.cap2.read()
        ret, image3 = self.cap3.read()
        ret, image4 = self.cap4.read()



    def open_camera(self):
        width = 640
        height = 360
        print(time.time() , 'open wrist camera')
        self.cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)  # top
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(3, width)
        self.cap.set(4, height)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap.set(cv2.CAP_PROP_FOCUS, 50)
        print(time.time() , 'open front camera')
        self.cap2 = cv2.VideoCapture(1,cv2.CAP_DSHOW)  # far
        self.cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap2.set(3, width)
        self.cap2.set(4, height)
        self.cap2.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap2.set(cv2.CAP_PROP_FOCUS, 20)
        print(time.time() , 'open right camera')
        self.cap3 = cv2.VideoCapture(2,cv2.CAP_DSHOW)  # top
        self.cap3.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap3.set(3, width)
        self.cap3.set(4, height)
        self.cap3.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap3.set(cv2.CAP_PROP_FOCUS, 10)
        print(time.time() , 'open top camera')
        self.cap4 = cv2.VideoCapture(3,cv2.CAP_DSHOW)  # far
        self.cap4.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap4.set(3, width)
        self.cap4.set(4, height)
        self.cap4.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap4.set(cv2.CAP_PROP_FOCUS,0)
        # while True:
        #     begin = time.time()
        #     ret, img = self.cap4.read()
        #     cv2.imshow('image0.jpg', img)
        #     # print(self.cap3.get(cv2.CAP_PROP_FOCUS))
        #     cv2.waitKey(1)


        #     print((time.time() - begin), self.cap4.get(cv2.CAP_PROP_FPS))
        #
        # cv2.imwrite('image0.jpg', image)
        # cv2.imwrite('image2.jpg', image2)
        # cv2.imwrite('image3.jpg', image3)
        # cv2.imwrite('image4.jpg', image4)

    def move_to(self, angles):
        # self.robot.enable()
        self.robot.movej(angles)
        # robot.movel([-149, 432, 17, -114,-17,-97,8])
        time.sleep(0.5)
        # Monitor the task status and wait until the MoveJ operation is completed
        while True:
            task_status = self.robot.get_task_status()
            # print(time.time(), task_status)
            # print(self.robot.get_robot_status())
            if task_status.get('type') is None:  # Check if task has concluded
                # print("MoveJ operation completed.")
                break
            else:
                if self.robot.get_robot_status()['cartesianPosition']['z'] < -39.8:
                    print(self.robot.get_robot_status()['cartesianPosition'])
                    break
            time.sleep(0.05)
        self.robot.stop()

    def move_abs(self, angles):
        # self.robot.enable()
        Script = """
                R:MOVEJ %s,0.8
                """ % ",".join(str(i) for i in angles)

        # Start executing the WScript and notify
        self.robot.wscript(Script, repeatCount=1)
        # print("Executing WScript. Expected to run once...:" + Script)
        time.sleep(0.5)

        # Monitor the task status and provide updates until the script execution concludes
        try:
            while True:
                # print(self.robot.get_robot_status())
                task_status = self.robot.get_task_status()
                # print(
                #     f"Task Name: {task_status.get('type')}, Task Progress: {task_status.get('progress')}")

                # Check if the WScript execution has ended
                if task_status.get('type') is None:
                    # print("WScript execution concluded.")
                    break
                else:
                    if self.robot.get_robot_status()['cartesianPosition']['z'] * 1000 < -31.8:
                        print(self.robot.get_robot_status()['cartesianPosition'])
                        self.robot.stop()
                        return 1
                time.sleep(0.05)  # Brief pause between status checks

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        self.robot.stop()
        print('z:', self.robot.get_robot_status()['cartesianPosition']['z'] * 1000)
        return 0

    # def move_abs(self, angles):
    #     self.robot.move_absolute_angles(angles)

    def close_gripper(self):
        self.gripper_status = Control.GRIPPER_CLOSE
        write_len = self.ser.write("#000P1400T1000!".encode('utf-8'))
        time.sleep(1)
        # self.gripper_status = Control.GRIPPER_CLOSE

    def open_gripper(self):
        print('open_gripper')
        self.gripper_status = Control.GRIPPER_OPEN
        write_len = self.ser.write("#000P0900T1000!".encode('utf-8'))
        time.sleep(1)
        # self.gripper_status = Control.GRIPPER_OPEN

    def close(self):
        self.robot.stop()
        self.robot.disable()

robot = Control()
