#!/bin/sh
JOBS_PATH=ogbn_proteins_jobs
LOGS_PATH=logs
for ENTRY in "${JOBS_PATH}"/*.sh; do
  chmod +x $ENTRY
  FILE_NAME="$(basename "$ENTRY")"
  echo $FILE_NAME
  /mnt/cephfs2/asr/users/ming.tu/software/kaldi/egs/wsj/s5/utils/queue.pl -q g.q -l gpu=2 $LOGS_PATH/$FILE_NAME.log $ENTRY &
  sleep 3
done