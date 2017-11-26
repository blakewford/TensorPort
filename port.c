#include <avr/io.h>

int clock_gettime(clockid_t clk_id, struct timespec *tp)
{
    return 0;
}

__attribute__ ((optimize(3)))
void delay(uint32_t milliseconds)
{
    uint32_t cycles = (milliseconds*2280);
    while(cycles--)
    {
        asm volatile("nop");
    }
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
