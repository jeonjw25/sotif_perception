import subprocess
import time
import signal
import os
import rospy

if __name__ == "__main__":
    dir_path = '/root/catkin_ws/src/yolov8_ros/bagfile'
    rospy.init_node('topic_checker', anonymous=True)
    
    for file_name in os.listdir(dir_path):
        if file_name.endswith('.bag'):
            for i in range(1,8):
                print(f"Start {file_name}:#{i}")

                full_file_path = os.path.join(dir_path, file_name)
                launch_command = f"roslaunch yolov8_ros yolov8_ros.launch bagfile:={file_name} iter:={i}"
                rosbag_command = f"rosbag play {full_file_path} --wait-for-subscribers"
                process1 = subprocess.Popen(launch_command, shell=True, preexec_fn=os.setsid)
                process2 = subprocess.Popen(rosbag_command, shell=True, preexec_fn=os.setsid)
                while process2.poll() is None:
                    time.sleep(1)

                print(f"{file_name}: #{i} end")
          
                os.killpg(os.getpgid(process1.pid), signal.SIGINT)
