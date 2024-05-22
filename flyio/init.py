#!/usr/bin/env python3
import subprocess
import os

def run_bash_command(cmd):
    subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    os.chdir('./projects')
    command = "git clone https://github.com/li-il-li/rl-enzyme-engineering.git"
    run_bash_command(command)
    os.chdir('./rl-enzyme-engineering')
    os.mkdir('./data')
    command = "rclone sync datasets:enzyme-rl ./data/"
    run_bash_command(command)
    
