// PTY relay owned by the app; no input/output is written to disk.
#include <util.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <poll.h>
#include <errno.h>
#include <stdlib.h>
static volatile sig_atomic_t child = -1;
static void stop(int) { if (child > 0) { kill(-child, SIGKILL); kill(child, SIGKILL); } }
static bool send_all(int fd,const char* p,size_t n) { while(n) { ssize_t w=write(fd,p,n);if(w<0 && errno==EINTR)continue;if(w<=0)return false;p+=w;n-=w;}return true; }
int main(int argc,char**argv) {
 if(argc!=2)return 64;
 signal(SIGTERM,stop);signal(SIGHUP,stop);signal(SIGINT,stop);signal(SIGPIPE,SIG_IGN);
 int master=-1;winsize size{};size.ws_row=24;size.ws_col=100;
 pid_t pid=forkpty(&master,nullptr,nullptr,&size);
 if(pid<0)return 70;
 if(pid==0){setenv("TERM","dumb",1);setenv("HISTFILE","/dev/null",1);execl("/bin/zsh","zsh","-f","-c",argv[1],(char*)nullptr);_exit(71);}
 child=pid;char buffer[8192];bool running=true;
 while(running){pollfd fds[2]={{master,POLLIN,0},{STDIN_FILENO,POLLIN,0}};int r=poll(fds,2,-1);if(r<0){if(errno==EINTR)continue;break;}
 if(fds[0].revents&(POLLIN|POLLHUP|POLLERR)){ssize_t n=read(master,buffer,sizeof buffer);if(n<=0)break;if(!send_all(STDOUT_FILENO,buffer,n))break;}
 if(fds[1].revents&(POLLIN|POLLHUP|POLLERR)){ssize_t n=read(STDIN_FILENO,buffer,sizeof buffer);if(n<=0)break;if(!send_all(master,buffer,n))break;}
 }
 close(master);int status=0;
 // Closing the controlling terminal hangs up any remaining foreground command.
 kill(-pid,SIGHUP);
 for(int i=0;i<100;++i){pid_t r=waitpid(pid,&status,WNOHANG);if(r==pid){child=-1;return WIFEXITED(status)?WEXITSTATUS(status):128+WTERMSIG(status);}usleep(10000);}
 stop(SIGTERM);waitpid(pid,&status,0);child=-1;return WIFEXITED(status)?WEXITSTATUS(status):128+WTERMSIG(status);
}
