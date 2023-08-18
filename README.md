This is a wgpu-based radix sort.

NOTES:

It spawns a window to help with debugging in NSight (seems to make certain
features work, at least in release mode). Use `run_headless()` to not spawn the
window.

Using multi-pass scatter made things worse. Using shared memory in morton coding
made no difference, although the # of long scoreboard issues went down. Coding
was shared-memory limited.

ping-pong between 2 shared memory buffers to increase speed of prefix sum?
