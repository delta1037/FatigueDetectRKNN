#!/bin/bash
init_use=0
seg=13

# 修改连接名称
if [ $init_use -eq 1 ];then
    nmcli con modify Wired\ connection\ 1 connection.id eth0
fi

# 配置静态IP地址
nmcli con mod eth0 ipv4.addresses 192.168.${seg}.${1}/24
nmcli con mod eth0 ipv4.gateway 192.168.${seg}.254
nmcli con mod eth0 ipv4.method manual
nmcli con up eth0
