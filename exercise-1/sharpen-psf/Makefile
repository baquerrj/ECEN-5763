INCLUDE_DIRS =
LIB_DIRS =
CC = gcc

CDEFS=
CFLAGS= -O3 -msse3 $(INCLUDE_DIRS) $(CDEFS)
LIBS=-lpthread

PRODUCT=sharpen_grid sharpen
#PRODUCT=sharpen

HFILES=
CFILES= sharpen_grid.c sharpen.c
#CFILES= sharpen.c

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	${PRODUCT}

clean:
	-rm -f *.o *.NEW *~
	-rm -f ${PRODUCT} ${DERIVED} ${GARBAGE}

sharpen:	sharpen.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o $(LIBS)


sharpen_grid:	sharpen_grid.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o $(LIBS)


depend:

.c.o:
	$(CC) $(CFLAGS) -c $<
