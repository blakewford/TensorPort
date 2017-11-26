#/bin/bash
avr-objcopy -Ielf32-avr -Oihex $2 /tmp/upload.hex
python reset.py $1
sleep 2
avrdude -v -c wiring -p atmega2560 -P$1 -Uflash:w:/tmp/upload.hex:i -D
