#!/bin/bash -e

# SSH
echo $AUTHORIZED_KEYS > /root/.ssh/authorized_keys

# Git
git config --global credential.helper '!f() { sleep 1; echo "username=${GIT_USER}"; echo "password=${GIT_PASSWORD}"; }; f'
git config --global user.name "$GIT_NAME"
git config --global user.email "$GIT_EMAIL"

# Make variables available for logged in user
{
echo "export GIT_USER=$GIT_USER"
echo "export GIT_PASSWORD=$GIT_PASSWORD"
echo "export RCLONE_CONFIG=/root/projects/rl-enzyme-engineering/rclone.conf"
echo "export CUDA_HOME=/usr/local/cuda"
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
echo "export PATH=$PATH:$CUDA_HOME/bin"
} >> /root/.bashrc

/usr/sbin/sshd
exec "$@"

# Keep container running after startup
tail -f /dev/null
