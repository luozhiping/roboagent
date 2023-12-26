import time
from STR400_SDK.str400 import STR400
import struct
import serial
import pymodbus
import numpy as np

def hex_to_int(hex_string):
    return struct.unpack('i', bytearray.fromhex(hex_string)[::-1])[0]
def hex_to_short(hex_string):
    if int(hex_string, 16) & (1 << 14):
        return struct.unpack('h', bytearray.fromhex(hex_string[4:])[::-1])[0] - 32767 + 1
    else:
        return struct.unpack('h', bytearray.fromhex(hex_string[4:])[::-1])[0]
def int_to_hex(int_number):
    return struct.pack('i', int_number)[::-1].hex()

class RobotModbus:
    READ_DATA = '00 00 00 01'
    DATA_ONE = '00 00 00 01'
    DATA_ZERO = '00 00 00 00'

    READ_CODE = '03'
    WRITE_CODE = '06'

    GET_ABS_POS_CODE = '00 15'
    GET_RLT_POS_CODE = '00 13'
    MOVE_TO_POS = '00 81'
    OPEN_SYSTEM = '00 10'
    SET_SPEED = '00 2E'
    SET_ASC = '00 27'
    SET_DESC = '00 28'
    STOP = '00 33'
    SPEED_MODE = '00 2F'

    PER_ANGLE = 32767 / 360 * 101
    PER_ANGLE_ABS = 32767 / 360

    def __init__(self):
        self.robot = serial.Serial("COM3", 2250000, bytesize=8, parity='N', stopbits=1, rtscts=False)

    def close(self):
        self.robot.close()

    def open_system(self):
        for i in range(1, 6):
            command = '%s %s %s %s' % ('0%s' % i, RobotModbus.WRITE_CODE, RobotModbus.OPEN_SYSTEM, RobotModbus.DATA_ONE)
            self.get_serial_result(command)
        self.set_speed()


    def close_system(self):
        for i in range(1, 6):
            command = '%s %s %s %s' % ('0%s' % i, RobotModbus.WRITE_CODE, RobotModbus.OPEN_SYSTEM, RobotModbus.DATA_ZERO)
            self.get_serial_result(command)

    def set_speed(self):
        for i in range(1, 6):
            command = '%s %s %s %s' % ('0%s' % i, RobotModbus.WRITE_CODE, RobotModbus.SET_SPEED, '00 00 25 e8')
            self.get_serial_result(command)
            command = '%s %s %s %s' % ('0%s' % i, RobotModbus.WRITE_CODE, RobotModbus.SET_ASC, '00 00 00 64')
            self.get_serial_result(command)
            command = '%s %s %s %s' % ('0%s' % i, RobotModbus.WRITE_CODE, RobotModbus.SET_DESC, '00 00 00 64')
            self.get_serial_result(command)

    def move_absolute_angles(self, angles):
        # 获取当前absolute
        for i in range(1, len(angles) + 1):
            machine = '0%s' % i
            angle = angles[i - 1]
            self.move_absolute_angle(machine, angle)

    def move_to_angles(self, angles):
        while True:
            current = self.get_current_angle()
            angles_offset = np.clip(np.array(angles) - np.array(current), -10, 10).astype(np.float32)
            print('move angles:', angles_offset)
            self.move_absolute_angles(list(angles_offset))
            tmp = self.get_current_angle()
            angles_offset = (np.abs(np.array(angles) - np.array(tmp)) < 1).all()
            if angles_offset:
                break
            time.sleep(1)

    def move_direct(self):
        # 获取当前absolute
        command = '%s %s %s %s' % ('0%s' % 1, RobotModbus.WRITE_CODE, RobotModbus.SPEED_MODE, '00 00 04 b0')
        self.get_serial_result(command)
        time.sleep(5)
        self.stop()

    def stop(self):
        command = '%s %s %s %s' % ('0%s' % 1, RobotModbus.WRITE_CODE, RobotModbus.STOP, '00 00 00 01')
        self.get_serial_result(command)

    def get_robot_status(self):
        return {'jointAngle': self.get_current_angle()}

    def move_absolute_angle(self, machine, angle):
        # 获取当前absolute
        current = self.get_relative_position(machine)
        des = int(hex_to_int(current) + RobotModbus.PER_ANGLE * angle)
        # print('current:', current, 'moveto:', int_to_hex(des))
        self.move_to_position(int_to_hex(des), machine)
        return des

    def get_absolute_position(self, machine='01'):
        command = '%s %s %s %s' % (machine, RobotModbus.READ_CODE, RobotModbus.GET_ABS_POS_CODE, RobotModbus.READ_DATA)
        # print(command)
        return self.get_serial_result(command)

    def get_current_angle(self):
        angles = []
        for i in range(1, 7):
            current = self.get_absolute_position('0%s' % i)
        # print(current, hex_to_short(current), hex_to_short(current) / RobotModbus.PER_ANGLE_ABS)
            angles.append(hex_to_short(current) / RobotModbus.PER_ANGLE_ABS)
        return angles

    def get_relative_position(self, machine='01'):
        command = '%s %s %s %s' % (machine, RobotModbus.READ_CODE, RobotModbus.GET_RLT_POS_CODE, RobotModbus.READ_DATA)
        # print(command)
        return self.get_serial_result(command)

    def move_to_position(self, position, machine='01'):
        # a,b,c,d = position[2:4], position[4:6], position[6:8],position[8:10]
        # if len(position) == 3:
        #     a, b, c, d = '00', '00', '00', '0' + position[-1]
        # elif len(position) == 4:
        #     a, b, c, d = '00', '00', '00', position[-2:]
        # elif len(position) == 5:
        #     a, b, c, d = '00', '00', '0' + position[-3], position[-2:]
        # elif len(position) == 6:
        #     a, b, c, d = '00', '00', position[-4:-2], position[-2:]
        # elif len(position) == 7:
        #     a, b, c, d = '00', '0' + position[-5], position[-4:-2], position[-2:]
        # elif len(position) == 8:
        #     a, b, c, d = '00', position[-6:-4], position[-4:-2], position[-2:]
        command = '%s %s %s %s' % (machine, RobotModbus.WRITE_CODE, RobotModbus.MOVE_TO_POS,
                                   ' '.join(position[i:i+2] for i in range(0, len(position), 2)))
        # print(command)
        self.get_serial_result(command)

    def get_serial_result(self, command):
        crc = hex(pymodbus.utilities.computeCRC(bytes.fromhex(command)))
        if len(crc) == 6:
            command = command + ' ' + crc[4:] + ' ' + crc[2:4]
        else:
            command = command + ' ' + crc[3:] + ' 0' + crc[2]
        self.robot.write(bytes.fromhex(command))
        return self.robot.read(10).hex()[8:16]


if __name__ == '__main__':

    robot = RobotModbus()
    robot.open_system()
    angle_before = robot.get_current_angle()
    # robot.move_direct()
    # robot.move_absolute_angles([10,2,0.8,1,-3])
    robot.move_to_angles([0, 0, 90, 0, 90])

    time.sleep(2)
    angle_after = robot.get_current_angle()
    print(np.array(angle_after) - np.array(angle_before))
    robot.close_system()
    robot.close()
# robot2 = STR400(host='localhost', port=8080)
# robot2.enable()
# for i in range(5):
#     print(robot2.get_robot_status())
#     time.sleep(1)
# robot2.disable()

# 00000056 1度  86
# 0000038f 10   911
# 00002393      9107