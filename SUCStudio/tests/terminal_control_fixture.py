"""Exercise control characters against the real app-owned PTY helper."""
import os
import select
import subprocess
import sys


def check(command, control, expected_status, expected_output=b''):
    p = subprocess.Popen([sys.argv[1], command], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        data = b''
        while b'READY' not in data:
            assert select.select([p.stdout], [], [], 3)[0], 'Prompt timed out'
            chunk = os.read(p.stdout.fileno(), 1024)
            assert chunk, 'PTY closed before prompt'
            data += chunk
        p.stdin.write(control)
        p.stdin.flush()
        p.wait(timeout=3)
        data += p.stdout.read()
        assert p.returncode == expected_status, (p.returncode, data)
        assert expected_output in data, data
    finally:
        if p.poll() is None:
            p.terminate()
            p.wait(timeout=3)
        p.stdin.close()
        p.stdout.close()


check("printf 'READY\\n'; sleep 90", b'\x03', 130)
check("printf 'READY\\n'; read value; printf 'EOF received\\n'", b'\x04', 0, b'EOF received')
print('PTY control tests passed: Ctrl-C exit 130 and Ctrl-D EOF')
