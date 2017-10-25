CC:=g++
DEBUG:=
EXTRA:=
#make CC=avr-g++ EXTRA="-fpermissive -Iport -include stdlib.h -include sys/types.h -include port.h -include avr/io.h -include string.h -mmcu=atmega1284 port.c -fno-threadsafe-statics -Wl,-u,vfprintf -lprintf_flt"
tensor:
	$(CC) $(DEBUG) -std=c++11 runtime_single_threaded_matmul.cc -o MatMul -I/home/blakewford/tensorflow -Ieigen $(EXTRA) -w

clean: MatMul
	rm $<
