#!/bin/sh

DURATION=${1:-10}
RATE=${2:-5}
TIMEOUT=${3:-5}
WORKERS=${4:-10}

vegeta attack -duration=${DURATION}s -rate=${RATE} -timeout=${TIMEOUT}s -workers=${WORKERS} -targets=./target.txt | vegeta report -type=text