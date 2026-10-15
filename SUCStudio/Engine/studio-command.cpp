// App-owned command runner: isolated process group permits bounded cancellation.
#include <unistd.h>
#include <stdio.h>
int main(int argc, char **argv) {
    if (argc != 2) return 64;
    if (getpgrp() != getpid() && setsid() < 0) { perror("setsid"); return 70; }
    execl("/bin/zsh", "zsh", "-c", argv[1], static_cast<char *>(nullptr));
    perror("exec"); return 71;
}
