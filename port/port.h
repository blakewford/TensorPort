#include <avr/io.h>

typedef int ssize_t;
namespace std
{
    typedef unsigned int size_t;
    typedef int ptrdiff_t;
}

void delay(uint32_t milliseconds);

#define CHAR_BIT 8
