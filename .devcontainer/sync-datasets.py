#!/usr/bin/env python3
import subprocess

def run_bash_command(cmd):
    subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    command = "rclone sync ./data/ datasets:enzyme-rl"
    run_bash_command(command)