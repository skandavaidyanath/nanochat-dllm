#!/bin/bash
# Deadline watchdog, runs ON the pod: terminates this pod after $1 hours through the Prime Intellect API, independent
# of the job chain, the ssh session and the laptop. launch_pod.sh starts it before anything else and verifies its PID.
# Needs ~/pod.env with PRIME_API_KEY and POD_ID. Log: ~/watchdog.log
HOURS="${1:?hours until forced termination}"
echo $$ > ~/watchdog.pid
echo "$(date -u +%FT%TZ) watchdog armed: terminate after ${HOURS} h (pid $$)" >> ~/watchdog.log
sleep $(( HOURS * 3600 ))
source ~/pod.env
echo "$(date -u +%FT%TZ) WATCHDOG TERMINATING pod $POD_ID" >> ~/watchdog.log
curl -s -X DELETE -H "Authorization: Bearer $PRIME_API_KEY" "https://api.primeintellect.ai/api/v1/pods/$POD_ID" >> ~/watchdog.log 2>&1
