#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BUFFER_SIZE 4

const char* pythonHeader = "\
import tensorflow as tf\n\
";

const char* pythonFooter = "\
  c = tf.matmul(a, b)\n\
sess = tf.Session()\n\
print(sess.run(c))\n\
";

void printDataset(int32_t rows, int32_t columns, const char* name)
{
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
    printf("], shape=[%d, %d], name='%s'", rows, columns, name);
}

int main(int argc, char** argv)
{
    // [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a'

    srand(time(NULL));

    int32_t data = 0;
    bool both = false;
    bool forPort = false;
    bool forProduct = false;
    if(argc == 1)
    {
        data = 1;
    }
    else if(argc != 3)
    {
        printf("usage: gen <num datasets> <target>\n");
        printf("target can be tensorport or tensorflow\n");
        return -1;
    }

    if(!strcmp(argv[2], "tensorport"))
    {
        forPort = true;
    }
    else if(!strcmp(argv[2], "tensorflow"))
    {
        forProduct = true;
        printf("%s", pythonHeader);
    }
    else if(!strcmp(argv[2], "both"))
    {
        both = true;
        printf("%s", pythonHeader);
    }
    else if(argc != 1)
    {
        printf("Invalid target type: %s\n", argv[2]);
        return -1;
    }

    if(forPort || forProduct || both)
    {
        data = atoi(argv[1]);
    }

    while(data--)
    {
        int32_t targets = both ? 2: 1;
        const int32_t rows = (rand()%BUFFER_SIZE) + 1;
        const int32_t columns = (rand()%BUFFER_SIZE) + 1;
        const int32_t columns2 = (rand()%BUFFER_SIZE) + 1;

        if(both)
        {
            forPort = true;
            forProduct = false;
        }

        while(targets--)
        {
            if(forPort)
            {
               printf("./MatMul \"");
            }
            if(forProduct)
            {
                printf("with tf.device('/cpu:0'):\n  a = tf.constant(");
            }
            printDataset(rows, columns, "a");
            if(forPort)
            {
                printf("\" \"");
            }
            else if(forProduct)
            {
                printf(")\n  b = tf.constant(");
            }
            else
            {
                printf("\n");
            }
            printDataset(columns, columns2, "b");
            if(forPort)
            {
                printf("\"\n");
            }
            else if(forProduct)
            {
                printf(")");
                printf("\n%s\n", pythonFooter);
            }
            else
            {
                printf("\n");
            }

            if(both)
            {
                forPort = false;
                forProduct = true;
            }
        }
    }

    return 0;
}
