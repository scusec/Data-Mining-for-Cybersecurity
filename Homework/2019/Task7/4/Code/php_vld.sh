#!/bin/sh

wget https://pecl.php.net/get/vld-0.14.0.tgz
tar -zxvf vld-0.14.0.tgz
cd vld-0.14.0

phpize
./configure
make && make install

echo "extension=vld.so" >> /etc/php.ini
