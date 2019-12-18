#!/bin/bash

PIDFILE="/home/ralph/project/Who_wants_to_die/Flask_server/process.pid"  # PID文件路径

if [ -f $PIDFILE ]; then
    PID=$(cat $PIDFILE)           # 将PID从文件中读取，并作为一个变量
    echo "Try kill $PID"
    kill -QUIT $PID
    echo "kill $PID"
    echo "Update Known_faces....Restart...." 
    
fi

nohup python3 torch_server_with_face_recognization.py > old_server.log 2>&1 &  echo $! > process.pid