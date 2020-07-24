#include <ctype.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <atomic>

#ifdef __LINUX__
#include <sys/sysinfo.h>
#endif

namespace xla
{
    void EigenMatVecF32(float* out, float* lhs, float* rhs, long long m, long long n, long long k, int transpose_lhs, int transpose_rhs)
    {
        assert(0);
    }

    void EigenMatVecF64(double* out, double* lhs, double* rhs, long long m, long long n, long long k, int transpose_lhs, int transpose_rhs)
    {
        assert(0);
    }
}

namespace std
{
    void swap(long long& A, long long& B)
    {
        long long C = A;
        A = B;
        B = C;
    }
}

#include "runtime_single_threaded_matmul.cc"

#define BUFFER_SIZE 16

//  Python parameters
//  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
//  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')

int32_t cachedArgc = 0;
char argvStorage[1024];
char* cachedArgv[64];

//  Default parameters
const char* gDefaults[] = {
    "[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='a', shape=[2, 3]",
    "[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b'",
    "[ 1.0, 5.0 ], shape=[2, 1], name='a'",
    "[ 8.0, 3.0, 5.0 ], shape=[1, 3], name='b'",
    "[ 4.0, 4.0 ], shape=[2, 1], name='a'",
    "[ 1.0, 3.0, 6.0 ], shape=[1, 3], name='b'",
    "[ 3.0 ], shape=[1, 1], name='a'",
    "[ 7.0 ], shape=[1, 1], name='b'",
    "[ 7.0, 9.0 ], shape=[2, 1], name='a'",
    "[ 4.0, 3.0, 3.0 ], shape=[1, 3], name='b'",
    "[ 0.0, 3.0, 3.0, 8.0, 1.0, 9.0 ], shape=[3, 2], name='a'",
    "[ 5.0, 8.0 ], shape=[2, 1], name='b'",
    "[ 9.0, 8.0, 5.0, 7.0 ], shape=[2, 2], name='a'",
    "[ 6.0, 9.0, 9.0, 1.0, 7.0, 6.0 ], shape=[2, 3], name='b'",
    "[ 3.0, 7.0, 0.0, 8.0 ], shape=[1, 4], name='a'",
    "[ 6.0, 9.0, 0.0, 6.0, 4.0, 6.0, 4.0, 7.0, 7.0, 0.0, 5.0, 0.0, 7.0, 0.0, 9.0, 8.0 ], shape=[4, 4], name='b'",
    "[ 5.0, 7.0, 6.0, 4.0 ], shape=[4, 1], name='a'",
    "[ 5.0 ], shape=[1, 1], name='b'",
    "[ 7.0, 4.0, 3.0, 5.0 ], shape=[1, 4], name='a'",
    "[ 0.0, 5.0, 7.0, 7.0, 9.0, 3.0, 3.0, 9.0, 0.0, 5.0, 4.0, 2.0, 5.0, 6.0, 3.0, 3.0 ], shape=[4, 4], name='b'",
    "[ 9.0, 0.0, 7.0, 5.0, 6.0, 9.0, 2.0, 9.0, 8.0, 6.0, 2.0, 5.0 ], shape=[3, 4], name='a'",
    "[ 9.0, 9.0, 3.0, 8.0, 9.0, 6.0, 1.0, 8.0, 9.0, 9.0, 4.0, 1.0, 4.0, 1.0, 5.0, 9.0 ], shape=[4, 4], name='b'",
    "[ 6.0, 9.0, 6.0, 1.0, 7.0, 7.0 ], shape=[2, 3], name='a'",
    "[ 5.0, 6.0, 8.0, 3.0, 0.0, 3.0, 2.0, 0.0, 6.0 ], shape=[3, 3], name='b'",
    "[ 4.0, 1.0 ], shape=[1, 2], name='a'",
    "[ 4.0, 3.0, 7.0, 7.0, 9.0, 0.0, 4.0, 8.0 ], shape=[2, 4], name='b'",
    "[ 5.0, 2.0, 4.0, 6.0, 9.0, 4.0, 3.0, 7.0, 2.0, 7.0, 9.0, 7.0, 1.0, 9.0, 6.0, 2.0 ], shape=[4, 4], name='a'",
    "[ 2.0, 3.0, 8.0, 5.0 ], shape=[4, 1], name='b'",
    "[ 6.0, 0.0, 3.0, 1.0 ], shape=[2, 2], name='a'",
    "[ 0.0, 3.0, 4.0, 6.0 ], shape=[2, 2], name='b'",
    "[ 7.0 ], shape=[1, 1], name='a'",
    "[ 7.0, 9.0 ], shape=[1, 2], name='b'",
    "[ 9.0 ], shape=[1, 1], name='a'",
    "[ 3.0, 0.0, 1.0, 5.0 ], shape=[1, 4], name='b'",
    "[ 1.0, 3.0, 3.0, 8.0 ], shape=[1, 4], name='a'",
    "[ 6.0, 8.0, 7.0, 8.0, 3.0, 8.0, 8.0, 6.0 ], shape=[4, 2], name='b'",
    "[ 0.0, 9.0, 4.0 ], shape=[1, 3], name='a'",
    "[ 7.0, 8.0, 6.0, 1.0, 1.0, 6.0, 6.0, 4.0, 7.0, 1.0, 2.0, 4.0 ], shape=[3, 4], name='b'",
    "[ 6.0, 4.0, 3.0, 6.0 ], shape=[4, 1], name='a'",
    "[ 1.0, 1.0, 9.0, 9.0 ], shape=[1, 4], name='b'",
    "[ 7.0, 5.0, 3.0, 9.0 ], shape=[4, 1], name='a'",
    "[ 9.0, 0.0 ], shape=[1, 2], name='b'",
    "[ 3.0, 3.0, 2.0, 7.0, 2.0, 5.0, 1.0, 6.0 ], shape=[2, 4], name='a'",
    "[ 6.0, 7.0, 5.0, 4.0, 4.0, 8.0, 2.0, 5.0, 1.0, 2.0, 7.0, 2.0, 2.0, 2.0, 0.0, 7.0 ], shape=[4, 4], name='b'",
    "[ 8.0, 0.0, 5.0, 4.0, 3.0, 1.0 ], shape=[3, 2], name='a'",
    "[ 6.0, 0.0, 3.0, 1.0, 4.0, 2.0 ], shape=[2, 3], name='b'",
    "[ 4.0, 5.0, 8.0, 7.0 ], shape=[1, 4], name='a'",
    "[ 3.0, 9.0, 1.0, 2.0, 4.0, 3.0, 4.0, 6.0, 2.0, 0.0, 7.0, 0.0, 1.0, 7.0, 7.0, 5.0 ], shape=[4, 4], name='b'",
    "[ 3.0, 2.0, 2.0, 7.0, 4.0, 2.0 ], shape=[2, 3], name='a'",
    "[ 9.0, 3.0, 9.0, 6.0, 1.0, 8.0, 9.0, 1.0, 9.0, 1.0, 7.0, 4.0 ], shape=[3, 4], name='b'",
    "[ 0.0, 0.0 ], shape=[1, 2], name='a'",
    "[ 8.0, 1.0 ], shape=[2, 1], name='b'",
    "[ 8.0, 4.0, 7.0, 2.0 ], shape=[2, 2], name='a'",
    "[ 8.0, 2.0, 1.0, 4.0, 6.0, 2.0 ], shape=[2, 3], name='b'",
    "[ 9.0, 5.0, 0.0, 2.0, 6.0, 4.0, 9.0, 2.0 ], shape=[2, 4], name='a'",
    "[ 2.0, 2.0, 8.0, 4.0, 2.0, 6.0, 6.0, 1.0, 4.0, 2.0, 0.0, 8.0, 2.0, 4.0, 6.0, 6.0 ], shape=[4, 4], name='b'",
    "[ 0.0, 7.0, 2.0, 1.0 ], shape=[4, 1], name='a'",
    "[ 8.0, 7.0, 3.0 ], shape=[1, 3], name='b'",
    "[ 1.0 ], shape=[1, 1], name='a'",
    "[ 8.0, 3.0, 3.0, 7.0 ], shape=[1, 4], name='b'",
    "[ 4.0, 9.0, 9.0, 8.0 ], shape=[1, 4], name='a'",
    "[ 1.0, 9.0, 0.0, 7.0, 5.0, 8.0, 6.0, 5.0 ], shape=[4, 2], name='b'",
    "[ 5.0, 9.0, 5.0, 2.0, 3.0, 6.0, 8.0, 4.0, 9.0, 7.0, 8.0, 5.0 ], shape=[4, 3], name='a'",
    "[ 6.0, 6.0, 2.0, 3.0, 0.0, 3.0, 3.0, 8.0, 6.0, 2.0, 1.0, 5.0 ], shape=[3, 4], name='b'",
    "[ 7.0, 4.0, 0.0, 2.0 ], shape=[1, 4], name='a'",
    "[ 9.0, 2.0, 8.0, 4.0, 7.0, 6.0, 4.0, 1.0, 5.0, 1.0, 9.0, 0.0 ], shape=[4, 3], name='b'",
    "[ 3.0, 7.0, 0.0, 8.0, 6.0, 7.0, 2.0, 9.0 ], shape=[4, 2], name='a'",
    "[ 4.0, 2.0, 0.0, 6.0 ], shape=[2, 2], name='b'",
    "[ 4.0, 6.0, 1.0, 4.0, 0.0, 8.0 ], shape=[3, 2], name='a'",
    "[ 2.0, 5.0, 1.0, 0.0 ], shape=[2, 2], name='b'",
    "[ 8.0, 8.0, 7.0, 3.0, 6.0, 0.0 ], shape=[3, 2], name='a'",
    "[ 3.0, 4.0 ], shape=[2, 1], name='b'",
    "[ 1.0, 0.0, 5.0, 0.0, 2.0, 2.0, 9.0, 8.0 ], shape=[4, 2], name='a'",
    "[ 9.0, 2.0, 5.0, 9.0, 2.0, 7.0, 4.0, 3.0 ], shape=[2, 4], name='b'",
    "[ 2.0, 5.0, 5.0, 1.0, 8.0, 1.0, 1.0, 1.0 ], shape=[2, 4], name='a'",
    "[ 5.0, 0.0, 8.0, 0.0, 4.0, 8.0, 7.0, 6.0, 2.0, 0.0, 7.0, 1.0 ], shape=[4, 3], name='b'",
    "[ 8.0, 1.0, 5.0, 5.0, 6.0, 5.0, 0.0, 3.0, 7.0, 5.0, 8.0, 0.0, 5.0, 9.0, 4.0, 8.0 ], shape=[4, 4], name='a'",
    "[ 6.0, 4.0, 6.0, 8.0 ], shape=[4, 1], name='b'",
    "[ 4.0, 9.0, 7.0, 1.0, 0.0, 6.0 ], shape=[3, 2], name='a'",
    "[ 2.0, 8.0, 7.0, 5.0, 4.0, 2.0, 2.0, 9.0 ], shape=[2, 4], name='b'",
    "[ 9.0, 5.0, 0.0, 6.0 ], shape=[1, 4], name='a'",
    "[ 4.0, 4.0, 4.0, 2.0 ], shape=[4, 1], name='b'",
    "[ 9.0, 9.0, 7.0, 6.0 ], shape=[4, 1], name='a'",
    "[ 1.0 ], shape=[1, 1], name='b'",
    "[ 3.0, 2.0, 2.0, 0.0, 7.0, 6.0, 2.0, 1.0 ], shape=[4, 2], name='a'",
    "[ 7.0, 8.0, 8.0, 5.0, 9.0, 3.0, 5.0, 5.0 ], shape=[2, 4], name='b'",
    "[ 1.0, 1.0, 4.0, 3.0, 0.0, 5.0, 3.0, 8.0, 6.0, 0.0, 8.0, 0.0 ], shape=[4, 3], name='a'",
    "[ 3.0, 0.0, 2.0, 4.0, 9.0, 0.0, 8.0, 3.0, 7.0, 7.0, 1.0, 4.0 ], shape=[3, 4], name='b'",
    "[ 4.0, 6.0 ], shape=[1, 2], name='a'",
    "[ 3.0, 5.0, 8.0, 4.0 ], shape=[2, 2], name='b'",
    "[ 7.0, 6.0, 5.0, 3.0, 8.0, 5.0, 5.0, 2.0, 5.0, 9.0, 8.0, 7.0, 9.0, 6.0, 2.0, 8.0 ], shape=[4, 4], name='a'",
    "[ 5.0, 5.0, 2.0, 4.0, 2.0, 6.0, 8.0, 9.0, 9.0, 5.0, 9.0, 6.0, 7.0, 4.0, 5.0, 4.0 ], shape=[4, 4], name='b'",
    "[ 1.0, 8.0, 7.0, 3.0, 5.0, 6.0 ], shape=[2, 3], name='a'",
    "[ 3.0, 4.0, 8.0, 0.0, 6.0, 6.0, 7.0, 2.0, 1.0, 1.0, 4.0, 7.0 ], shape=[3, 4], name='b'",
    "[ 9.0, 4.0, 6.0, 6.0, 9.0, 1.0, 2.0, 2.0 ], shape=[4, 2], name='a'",
    "[ 6.0, 3.0, 5.0, 4.0, 0.0, 1.0 ], shape=[2, 3], name='b'",
};

enum parse_state
{
    CONSTANTS,
    SHAPE,
    NAME,
    UNKNOWN
};

struct param
{
    float value[BUFFER_SIZE];
    float shape[BUFFER_SIZE];
    char name[BUFFER_SIZE];
};

void TensorPort(const param& A, const param& B, float* C)
{
    assert(A.shape[1] == B.shape[0]);

    MatMul(NULL, C, (float*)A.value, (float*)B.value, A.shape[0], B.shape[1], A.shape[1], true, true);

    int32_t count = 0;
    int32_t index = 0;
    int32_t rows = A.shape[0];
    const int32_t columns = B.shape[1];

    int32_t currentColumn = columns;
    char buffer[BUFFER_SIZE];
//    write(1, "[[ ", 3);
    while(rows--)
    {
        while(currentColumn--)
        {
            memset(buffer, '\0', BUFFER_SIZE);
            sprintf(buffer, "%.2f ", C[index]);
            index += A.shape[0];
//            write(1, buffer, strlen(buffer));
        }
//        if(rows != 0)
//            write(1, "]\n [ ", 5);
//        else
//            write(1, "]", 1);
        currentColumn = columns;
        count++;
        index = count;
    }
//    write(1, "]\n", 2);
}

void parseEntry(const char* cursor, param& parameter, int32_t& valueSize, int32_t& shapeSize)
{
    valueSize = 0;
    shapeSize = 0;

    int32_t i = 0;
    bool nameOpen = false;
    char buffer[BUFFER_SIZE];
    memset(buffer, '\0', BUFFER_SIZE);

    parse_state state = UNKNOWN;

    while(*cursor != '\0')
    {
        char current = *cursor;
        switch(current)
        {
            case '[':
                if(state == UNKNOWN)
                {
                    state = CONSTANTS; // Cannot have non-keyword args after keyword args, so default to non-keyword.
                }
                break;
            case ',':
                switch(state)
                {
                    case CONSTANTS:
                        parameter.value[valueSize++] = strtod(buffer, NULL);
                        break;
                    case SHAPE:
                        parameter.shape[shapeSize++] = strtod(buffer, NULL);
                        break;
                }
                memset(buffer, '\0', BUFFER_SIZE);
                i = 0;
                break;
            case '=':
                {
                    char* front = buffer;
                    while(isspace(*front))
                        front++;

                    if(!strcmp(front, "shape"))
                    {
                        state = SHAPE;
                    }
                    else if(!strcmp(front, "name"))
                    {
                        state = NAME;
                    }
                    memset(buffer, '\0', BUFFER_SIZE);
                    i = 0;
                }
                break;
            case '\'':
                if(state == NAME && !nameOpen)
                {
                    nameOpen = true;
                }
                else
                {
                    strcpy(parameter.name, buffer);
                    memset(buffer, '\0', BUFFER_SIZE);
                    i = 0;
                    nameOpen = false;
                }
                break;
            case '.':
                if(state == CONSTANTS)
                {
                    buffer[i++] = current;
                }
                break;
            case ']':
                switch(state)
                {
                    case CONSTANTS:
                        parameter.value[valueSize++] = strtod(buffer, NULL);
                        break;
                    case SHAPE:
                        parameter.shape[shapeSize++] = strtod(buffer, NULL);
                        break;
                    default:
                        assert(0);
                }
                memset(buffer, '\0', BUFFER_SIZE);
                i = 0;
                state = UNKNOWN;
                break;
            default:
                if(state == CONSTANTS || state == SHAPE)
                {
                    if(isdigit(current))
                    {
                        buffer[i++] = current;
                    }
                }
                else if(state == UNKNOWN || state == NAME)
                {
                    buffer[i++] = current; //Parsing the name of the state
                }
                break;
        }
        cursor++;
    }
}

std::atomic<int> gNextCore(0);

void* iteration(void*)
{
    const pthread_t self = pthread_self();

    int core = gNextCore.fetch_add(1);

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    pthread_setaffinity_np(self, sizeof(cpu_set_t), &cpuset);

    int32_t loops = 200000; // ~5:20 seconds, ATmega 2560
    while(loops--)
    {
        int i = 0;
        const int count = sizeof(gDefaults)/sizeof(const char*);
        while(i < count)
        {
            param A, B;
            int32_t valueSize, shapeSize;
            parseEntry(gDefaults[i++], A, valueSize, shapeSize);
            parseEntry(gDefaults[i++], B, valueSize, shapeSize);
            float C[(int32_t)(A.shape[0]*B.shape[1])];
            TensorPort(A, B, C);
        }
    }

    return nullptr;
}

#ifndef __LINUX__
int get_nprocs()
{
    return 8;
}
#endif

int main(int argc, char** argv)
{
#ifdef __AVR__
    //Set as output pin
    DDRB = _BV(7);
    //Set LED
    PORTB = 0x80;
    delay(5000);
    //Clear LED
    PORTB = 0x00;
#else
    cachedArgc = argc;
    char* storagePointer = argvStorage;
    while(argc--)
    {
        cachedArgv[argc] = storagePointer;
        int32_t length = strlen(argv[argc]);
        strcat(storagePointer, argv[argc]);
        storagePointer+=(length+1);
    }
#endif

    const int processors = get_nprocs();
    pthread_t thread[processors];

    int count = processors;
    while(count--)
    {
        pthread_create(&thread[count], nullptr, iteration, nullptr);
    }

    count = processors;
    while(count--)
    {
        pthread_join(thread[count], nullptr);
    }

#ifdef __AVR__
    PORTB = 0x80;
#endif

    write(1, "\nDone\n", 6);

    return 0;
}
