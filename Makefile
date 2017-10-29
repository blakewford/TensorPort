CC:=g++
DEBUG:=
EXTRA:=
#make CC=avr-g++ EXTRA="-fpermissive -Iport -include stdlib.h -include sys/types.h -include port.h -include avr/io.h -include string.h -mmcu=atmega1284 port.c -fno-threadsafe-statics -Wl,-u,vfprintf -lprintf_flt -Os"
tensor:
	$(CC) $(DEBUG) -std=c++11 main.cpp -o MatMul -I/home/blakewford/tensorflow -Ieigen $(EXTRA) -w

gen: gen.cpp
	$(CC) gen.cpp -o gen

clean: MatMul
	rm $<
