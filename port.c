#include <avr/io.h>

int clock_gettime(clockid_t clk_id, struct timespec *tp)
{
    return 0;
}

ssize_t write(int fd, const void *buf, size_t count)
{
    char* c = (const char*)buf;
    while(count--)
    {
        /* Wait for empty transmit buffer */
        while ( !( UCSR1A & (1<<UDRE1)) )
            ;
        UDR0 = *c;
        c++;
    }

    return 0;
}
