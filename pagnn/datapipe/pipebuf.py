"""

Source: https://github.com/wal-e/wal-e/blob/master/wal_e/pipebuf.py
"""
import fcntl

PIPE_BUF_BYTES = None
OS_PIPE_SZ = None


def _configure_buffer_sizes():
    """Set up module globals controlling buffer sizes"""
    global PIPE_BUF_BYTES
    global OS_PIPE_SZ

    PIPE_BUF_BYTES = 65536
    OS_PIPE_SZ = None

    # Teach the 'fcntl' module about 'F_SETPIPE_SZ', which is a Linux-ism,
    # but a good one that can drastically reduce the number of syscalls
    # when dealing with high-throughput pipes.
    if not hasattr(fcntl, 'F_SETPIPE_SZ'):
        import platform

        if platform.system() == 'Linux':
            fcntl.F_SETPIPE_SZ = 1031

    # If Linux procfs (or something that looks like it) exposes its
    # maximum F_SETPIPE_SZ, adjust the default buffer sizes.
    try:
        with open('/proc/sys/fs/pipe-max-size', 'r') as f:
            # Figure out OS pipe size, but in case it is unusually large
            # or small restrain it to sensible values.
            OS_PIPE_SZ = min(int(f.read()), 50 * 1024 * 1024)
            PIPE_BUF_BYTES = max(OS_PIPE_SZ, PIPE_BUF_BYTES)
    except Exception:
        pass


_configure_buffer_sizes()


def set_buf_size(fd):
    """Set up os pipe buffer size, if applicable"""
    if OS_PIPE_SZ and hasattr(fcntl, 'F_SETPIPE_SZ'):
        fcntl.fcntl(fd, fcntl.F_SETPIPE_SZ, OS_PIPE_SZ)
