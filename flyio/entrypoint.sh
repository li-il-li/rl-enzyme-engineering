#!/bin/bash -e
echo $AUTHORIZED_KEYS > /root/.ssh/authorized_keys
git config --global credential.helper '!f() { sleep 1; echo "username=${GIT_USER}"; echo "password=${GIT_PASSWORD}"; }; f'
git config --global user.name "$GIT_NAME"
git config --global user.email "$GIT_EMAIL"

{
echo "export GIT_USER=$GIT_USER"
echo "export GIT_PASSWORD=$GIT_PASSWORD"
echo "export CUDA_HOME=/usr/local/cuda"
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
echo "export PATH=$PATH:$CUDA_HOME/bin"
} >> /root/.bashrc

# Check if repo is already on disk and if not clone it
REPO_URL="https://github.com/li-il-li/rl-enzyme-engineering.git"
REPO_NAME=$(basename -s .git "$REPO_URL")
if [ -d "/root/projects/$REPO_NAME" ]; then
    echo "Repository already exists!"
else
    git clone "$REPO_URL" "/root/projects/$REPO_NAME"
fi

/usr/sbin/sshd
exec "$@"

tail -f /dev/null
