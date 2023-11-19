#ifndef __BITMAP_H__
#define __BITMAP_H__

/* BMP 对象的类型定义 */
typedef struct {
    int   width;   /* 宽度 */
    int   height;  /* 高度 */
    int   stride;  /* 行字节数 */
    void *pdata;   /* 指向数据 */
} BMP;

int   bmp_load(BMP *pb, char *file);
void  bmp_free(BMP *pb);
float bmp_getpixel(BMP *pb, int i);

#endif


