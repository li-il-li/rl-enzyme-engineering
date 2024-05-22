#!/bin/bash -e
echo $AUTHORIZED_KEYS > /root/.ssh/authorized_keys
git config --global credential.helper '!f() { sleep 1; echo "username=${GIT_USER}"; echo "password=${GIT_PASSWORD}"; }; f'

{
echo "export GIT_USER=$GIT_USER"
echo "export GIT_PASSWORD=$GIT_PASSWORD"
echo "export RCLONE_CONFIG=/root/projects/rl-enzyme-engineering/rclone.conf"
} >> /root/.bashrc


/usr/sbin/sshd
exec "$@"

tail -f /dev/null