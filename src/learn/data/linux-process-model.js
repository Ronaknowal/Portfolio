export const processActions = {
  start: { label: "Start child", command: "sleep 60 &\npid=$!", from: ["ready"], to: "running", note: "The shell starts a child and keeps accepting commands. $! records that child's PID. Running here means alive and not stopped; sleep is normally waiting on a timer, not using a CPU continuously." },
  stop: { label: "Pause child", command: 'kill -STOP "$pid"', from: ["running"], to: "stopped", note: "SIGSTOP suspends the child. Its PID and memory still exist. It has not exited, and pausing does not undo any earlier work." },
  resume: { label: "Continue child", command: 'kill -CONT "$pid"', from: ["stopped"], to: "running", note: "SIGCONT lets the stopped child continue. This resumes the same process; it does not start a new copy." },
  terminate: { label: "End child", command: 'kill -TERM "$pid"', from: ["running"], to: "exited", note: "This sleep process uses SIGTERM's default termination behavior. Other programs may handle or ignore TERM. The child has ended; its exit status is available to the parent." },
  wait: { label: "Collect exit status", command: 'wait "$pid"\nprintf \'status=%s\\n\' "$?"', from: ["exited"], to: "collected", note: "Bash's wait returns 143 here (128 + signal 15). Bash may already have reaped the OS process and retained its status before you type wait. Collecting a result is different from sending a signal." },
};

export function transitionProcess(state, action) {
  const next = processActions[action];
  return next?.from.includes(state) ? next.to : state;
}
