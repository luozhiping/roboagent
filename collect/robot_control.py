from STR400_SDK.str400 import STR400
import time
import serial
import cv2
from threading import Thread
import pickle
from _datetime import datetime
import os
import numpy as np


# 1.舵机位置#000PRAD!
# 2.sleep 1s 后开始记录
class Control:

    BEGIN_ANGLE = [-10, 0, 90, 0, 50, 0]
    END_ANGLE = [0, 0, 90, 0, 90, 0]
    GRIPPER_OPEN = 0
    GRIPPER_CLOSE = 1
    GRIPPER_OPENING = 2
    GRIPPER_CLOSING = 3
    def __init__(self):
        self.robot = STR400(host='localhost', port=8080)
        self.ser = serial.Serial("COM6", 115200)  # 打开COM17，将波特率配置为115200，其余参数使用默认值
        self.folder_name = "%s" % (datetime.now().strftime("%m_%d"))
        os.makedirs("output/%s" % self.folder_name, exist_ok=True)

        self.job = 'put red block to pan.'

        self.pos_start = None
        self.pos_end = None
        self.robot_status = None
        self.gripper_command = None
        self.recording = False
        self.data = []
        self.gripper_status = Control.GRIPPER_CLOSE
        self.gripper_pos = 0
        t = Thread(target=self.get_status)
        t.start()
        # time.sleep(1)
        self.open_gripper()
        # time.sleep(1)
        # self.close_gripper()
        self.open_camera()
        self.begin_mark()


    def get_status(self):

        # self.robot.enable()
        while True:
            if self.gripper_command == 'open':
                self.gripper_status = Control.GRIPPER_OPEN
                write_len = self.ser.write("#000P0900T1000!".encode('utf-8'))
                self.gripper_command = None
            elif self.gripper_command == 'close':
                self.gripper_status = Control.GRIPPER_CLOSE
                write_len = self.ser.write("#000P1350T1000!".encode('utf-8'))
                self.gripper_command = None
            else:
            # self.robot_status = self.robot.get_robot_status()
                self.ser.write("#000PRAD!".encode('utf-8'))
                self.gripper_pos = self.ser.read(10).decode('utf-8')[-5:-1]
                # print(self.gripper_pos)
            time.sleep(0.02)

    def begin_mark(self):
        # {'jointEnabled': [1, 1, 1, 1, 1, 1], 'jointAngle': [-0.021973326822717354, 0.0988799707022309, 0.054933317056794946, -0.021973326822717354, 90.29938657795952, 0.02197332682271798],
        # 'cartesianPosition': {'x': -0.0001137956487933526, 'y': 0.14872586823021186, 'z': 0.5075285240957139, 'roll': -90.45319998771501, 'pitch': -0.04394566628930179, 'yaw': -90.02226194664331},
        # 'encoderAbsolutePositions': [0, 0, 0, 0, 0, 0], 'jointTemperature': [0, 0, 0, 0, 0, 0], 'jointSpeed': [0, 0, 0, 0, 0, 0],
        # 'jointErrorCode': [0, 0, 0, 0, 0, 0], 'jointRunning': [0, 0, 0, 0, 0, 0], 'jointCurrent': [0, 0, 0, 0, 0, 0]}
        self.robot.stop()
        self.robot.disable()
        while True:
            command = input('等待指令')
            if command == 'm':
                self.pos_start = self.robot.get_robot_status()['jointAngle']
                print('pos_start:' , self.robot.get_robot_status())
            elif command == 'n':
                self.pos_end = self.robot.get_robot_status()['jointAngle']
                print('pos_end:' , self.robot.get_robot_status())
            elif command == 's':
                print('begin move')
                for i in range(7):
                    self.begin_move()
                self.move_to([0, 0, 0, 0, 90, 0] + [2])
                self.close()
            elif command == 'x':
                self.close()


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

        ret, image = self.cap.read()
        ret, image2 = self.cap2.read()
        ret, image3 = self.cap3.read()
        ret, image4 = self.cap4.read()
        #     print((time.time() - begin), self.cap4.get(cv2.CAP_PROP_FPS))
        #
        # cv2.imwrite('image0.jpg', image)
        # cv2.imwrite('image2.jpg', image2)
        # cv2.imwrite('image3.jpg', image3)
        # cv2.imwrite('image4.jpg', image4)

    def move_to(self, angles):
        # print(time.time(), ' begin enable')
        # self.robot.enable()
        # print(time.time(), ' end enable')
        self.robot.movej(angles)
        # print(time.time(), 'submit move')
        # robot.movel([-149, 432, 17, -114,-17,-97,8])
        time.sleep(0.5)
        # Monitor the task status and wait until the MoveJ operation is completed
        # print("Monitoring task status until the MoveJ operation is completed...")
        while True:
            task_status = self.robot.get_task_status()
            # print(time.time(), task_status)
            # print(self.robot.get_robot_status())
            if task_status.get('type') is None:  # Check if task has concluded
                # print("MoveJ operation completed.")
                # break
                # print(time.time(), 'stop move')
                self.robot.stop()
                # print(time.time(), 'end move')
                return
            # else:
                # self.robot.get_robot_status()['']
            time.sleep(0.02)

    def move_to2(self, angles):

        Script = """
        A:MOVEJ %s
        """ % ",".join(str(i) for i in angles)

        # Start executing the WScript and notify
        self.robot.wscript(Script, repeatCount=1)
        print("Executing WScript. Expected to run once...:" + Script)
        time.sleep(0.5)

        # Monitor the task status and provide updates until the script execution concludes
        try:
            while True:
                task_status = self.robot.get_task_status()
                print(
                    f"Task Name: {task_status.get('type')}, Task Progress: {task_status.get('progress')}")

                # Check if the WScript execution has ended
                if task_status.get('type') is None:
                    print("WScript execution concluded.")
                    break

                time.sleep(0.2)  # Brief pause between status checks

        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def close_gripper(self):
        self.gripper_status = Control.GRIPPER_CLOSE
        # write_len = self.ser.write("#000P1400T1000!".encode('utf-8'))
        self.gripper_command = 'close'
        time.sleep(1)
        # self.gripper_status = Control.GRIPPER_CLOSE

    def open_gripper(self):
        print('open_gripper')
        self.gripper_status = Control.GRIPPER_OPEN
        self.gripper_command = 'open'
        # write_len = self.ser.write("#000P0900T1000!".encode('utf-8'))
        time.sleep(1)
        # self.gripper_status = Control.GRIPPER_OPEN

    def begin_move(self):
        self.robot.enable()

        gas = np.array(Control.BEGIN_ANGLE) + np.random.normal(loc=0.0, scale=6, size=6)
        print(list(gas))
        self.move_to(list(gas) + [2])
        # self.open_gripper()

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

        self.recording = True
        t = Thread(target=self.record)
        t.start()
        self.move_to(self.pos_start + [3])
        # self.open_gripper()
        self.close_gripper()
        # self.move_to(self.pos_end + [2])
        # time.sleep(0.3)
        self.recording = False
        self.open_gripper()
        # self.move_to([0, 0, 0, 0, 90, 0] + [2])
        # self.open_gripper()
        # self.close()
        # command = input('是否保存?')
        # if command == 'p':
        epsoide_name = datetime.now().strftime("%m_%d_%H_%M_%S")
        pickle.dump(self.data, open('output/%s/%s.pkl' % (self.folder_name, epsoide_name), 'wb'))
        print('save to %s' % 'output/%s/%s.pkl' % (self.folder_name, epsoide_name))
    def record(self):
        self.data.clear()
        while self.recording:
            begin = time.time()
            ret, image = self.cap.read()
            ret, image2 = self.cap2.read()
            ret, image3 = self.cap3.read()
            ret, image4 = self.cap4.read()
            retval, buffer = cv2.imencode('.jpg', image)
            retval, buffer2 = cv2.imencode('.jpg', image2)
            retval, buffer3 = cv2.imencode('.jpg', image3)
            retval, buffer4 = cv2.imencode('.jpg', image4)

            self.data.append({'robot_status': self.robot.get_robot_status(),
                              'wrist': buffer.tobytes(),
                              'front': buffer2.tobytes(),
                              'right': buffer3.tobytes(),
                              'top': buffer4.tobytes(),
                              'gripper_status': self.gripper_status,
                              'gripper_pos': self.gripper_pos,
                              'job': self.job
                              })
            print('record:', (time.time() - begin))
            time.sleep(0.1 - (time.time() - begin))

    def close(self):
        self.robot.stop()
        self.robot.disable()

robot = Control()
