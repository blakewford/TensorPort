#ifndef _STRUCT_TIMESPEC
#define _STRUCT_TIMESPEC
struct timespec {
  time_t tv_sec;
  long	 tv_nsec;
};

typedef struct tm {
  int tm_sec;
  int tm_min;
  int tm_hour;
  int tm_mday;
  int tm_mon;
  int tm_year;
  int tm_wday;
  int tm_yday;
  int tm_isdst;
} tm;

int clock_gettime(clockid_t clk_id, struct timespec *tp);

#endif

#define CLOCK_REALTIME 0
#define CLOCK_MONOTONIC 1
#define CLOCK_PROCESS_CPUTIME_ID 2
#define CLOCK_THREAD_CPUTIME_ID 3
