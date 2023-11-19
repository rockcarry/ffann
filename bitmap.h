#ifndef __BITMAP_H__
#define __BITMAP_H__

typedef struct {
    int   width;
    int   height;
    int   stride;
    void *pdata;
} BMP;

int   bmp_load(BMP *pb, char *file);
void  bmp_free(BMP *pb);
float bmp_getpixel(BMP *pb, int i);

#endif

