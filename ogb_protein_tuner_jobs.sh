#!/bin/sh
JOBS_PATH=ogbn_proteins_jobs
LOGS_PATH=logs
for ENTRY in "${JOBS_PATH}"/*.sh; do
  chmod +x $ENTRY
  FILE_NAME="$(basename "$ENTRY")"
  echo $FILE_NAME
  /home/snap/Documents/guangtaowang/GraphDiffusionForOGB/ogbnodeclassification/queue.pl -l gpu=1 $LOGS_PATH/$FILE_NAME.log $ENTRY &
  sleep 3
done