CC:=g++
DEBUG:=
EXTRA:=
#make CC=avr-g++ EXTRA="-fpermissive -Iport -include stdlib.h -include sys/types.h -include port.h -include avr/io.h -include string.h -mmcu=atmega2560 port.c -fno-threadsafe-statics -Wl,-u,vfprintf -lprintf_flt -Os"
tensor:
	$(CC) $(DEBUG) -std=c++11 main.cpp -o MatMul -I/home/blakewford/tensorflow -Ieigen $(EXTRA) -w

gen: gen.cpp
	$(CC) -g gen.cpp -o gen

clean: MatMul
	rm $<
