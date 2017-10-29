#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#define BUFFER_SIZE 4

int main(int argc, char** argv)
{
    // [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a'

    srand(time(NULL));
    const int32_t rows = (rand()%BUFFER_SIZE) + 1;
    const int32_t columns = (rand()%BUFFER_SIZE) + 1;

    int32_t rowCount = rows;
    printf("[ ");
    while(rowCount--)
    {
        int32_t count = columns;
        while(count--)
        {
            printf("%.1f", (float)(rand()%10));
            if(count != 0)
            {
                printf(", ");
            }
            else if(rowCount != 0)
            {
                printf(", ");
            }
            else
            {
                printf(" ");
            }
        }
    }
    printf("], shape=[%d, %d], name='a'\n", rows, columns);

    int32_t columns2 = (rand()%BUFFER_SIZE) + 1;

    rowCount = columns;
    printf("[ ");
    while(rowCount--)
    {
        int32_t count = columns2;
        while(count--)
        {
            printf("%.1f", (float)(rand()%10));
            if(count != 0)
            {
                printf(", ");
            }
            else if(rowCount != 0)
            {
                printf(", ");
            }
            else
            {
                printf(" ");
            }
        }
    }
    printf("], shape=[%d, %d], name='b'\n", columns, columns2);

    return 0;
}
