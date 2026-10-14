#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
uint64_t x;
uint64_t next() {
	uint64_t z = (x += 0x9e3779b97f4a7c15);
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
	return z ^ (z >> 31);
}
int main(int c,char**v){x=strtoull(v[1],0,10);for(int i=0;i<5;i++)printf("%016llx\n",(unsigned long long)next());}
