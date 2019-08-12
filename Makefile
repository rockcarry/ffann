CC      = gcc
CCFLAGS = -Wall -Os -lm
AR      = ar

OBJS = \
    matrix.o \
    bitmap.o \
    samples.o\
    ffann.o

EXES = \
    example1.exe \
    example2.exe \
    example3.exe

LIB = libffaan.a

# ±‡“ÎπÊ‘Ú
all : $(LIB) $(EXES)

%.o : %.c %.h
	$(CC) $(CCFLAGS) -c $<

$(LIB) : $(OBJS)
	$(AR) rcs $@ $(OBJS)

%.exe : %.c $(LIB)
	$(CC) -o $@ $< $(LIB)

clean :
	rm -f *.o
	rm -f *.a
	rm -f *.exe


