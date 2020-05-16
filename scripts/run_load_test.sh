#!/bin/sh

DURATION=${1:-10}
RATE=${2:-5}

vegeta attack -duration=${DURATION}s -rate=${RATE} -targets=./target.txt | vegeta report -type=text