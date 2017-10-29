#include <ctype.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <unistd.h>

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
    void swap(long long&, long long&)
    {
        assert(0);
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
    "[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b'"
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

    MatMul(NULL, C, (float*)A.value, (float*)B.value, A.shape[0], B.shape[1], A.shape[1], false, false);

    int32_t count = 0;
    const int32_t rows = A.shape[0];
    int32_t columns = B.shape[1];

    int32_t currentRow = rows;
    char buffer[BUFFER_SIZE];
    write(1, "[[ ", 3);
    while(columns--)
    {
        while(currentRow--)
        {
            memset(buffer, '\0', BUFFER_SIZE);
            sprintf(buffer, "%.2f ", C[count++]);
            write(1, buffer, strlen(buffer));
        }
        if(columns != 0)
            write(1, "]\n [ ", 5);
        else
            write(1, "]", 1);      
        currentRow = rows;
    }
    write(1, "]\n", 2);
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

int main(int argc, char** argv)
{
    cachedArgc = argc;
    char* storagePointer = argvStorage;
    while(argc--)
    {
        cachedArgv[argc] = storagePointer;
        int32_t length = strlen(argv[argc]);
        strcat(storagePointer, argv[argc]);
        storagePointer+=(length+1);
    }

    param A, B;
    int32_t valueSize, shapeSize;
    parseEntry(cachedArgc > 1 ? cachedArgv[1]: gDefaults[0], A, valueSize, shapeSize);
    parseEntry(cachedArgc > 2 ? cachedArgv[2]: gDefaults[1], B, valueSize, shapeSize);

    float C[valueSize];

    TensorPort(A, B, C);

    return 0;
}