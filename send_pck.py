from scapy.all import *
import sys
from time import sleep

while True:
    send(IP(dst='192.168.8.1')/UDP(dport=55555)/"hello")
    sleep(0.5)
    #p=sr1(IP(dst='192.168.8.1')/ICMP()/"hello1")
    #p.show()

#sr(IP(dst='192.168.8.183')/ICMP()/"hello")
#sr(IP(dst='192.168.8.183')/ICMP()/"hello")
#send(IP(dst='192.168.8.183')/UDP(dport=55555))
#send(IP(dst='192.168.8.183')/UDP(dport=55555))
#send(IP(dst='192.168.8.183')/UDP(dport=55555))
