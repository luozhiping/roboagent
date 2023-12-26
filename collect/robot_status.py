from STR400_SDK.str400 import STR400
import time
robot = STR400(host='localhost', port=8080)
while True:
    print(robot.get_robot_status()['cartesianPosition']['z'] * 1000)
    time.sleep(1)
