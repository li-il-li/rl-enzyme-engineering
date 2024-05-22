#!/bin/bash -e
echo $AUTHORIZED_KEYS > /root/.ssh/authorized_keys

/usr/sbin/sshd
exec "$@"

tail -f /dev/null