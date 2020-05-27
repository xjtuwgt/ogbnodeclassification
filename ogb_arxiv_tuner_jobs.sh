#!/bin/sh
JOBS_PATH=ogbn_arxiv_jobs
LOGS_PATH=logs
for ENTRY in "${JOBS_PATH}"/*.sh; do
  chmod +x $ENTRY
  FILE_NAME="$(basename "$ENTRY")"
  echo $FILE_NAME
  /home/snap/Documents/guangtaowang/GraphDiffusionForOGB/ogbnodeclassification/queue.pl -q g.q -l gpu=2 $LOGS_PATH/$FILE_NAME.log $ENTRY &
  sleep 3
done