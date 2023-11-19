#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "utils.h"
#include "bitmap.h"

#pragma pack(1)
typedef struct {
    uint16_t  bfType;
    uint32_t  bfSize;
    uint16_t  bfReserved1;
    uint16_t  bfReserved2;
    uint32_t  bfOffBits;
    uint32_t  biSize;
    uint32_t  biWidth;
    uint32_t  biHeight;
    uint16_t  biPlanes;
    uint16_t  biBitCount;
    uint32_t  biCompression;
    uint32_t  biSizeImage;
    uint32_t  biXPelsPerMeter;
    uint32_t  biYPelsPerMeter;
    uint32_t  biClrUsed;
    uint32_t  biClrImportant;
} BMPFILEHEADER;
#pragma pack()

int bmp_load(BMP *pb, char *file)
{
    BMPFILEHEADER header = {0};
    FILE         *fp     = NULL;
    uint8_t      *pdata  = NULL;
    int           ret, i;

    (void)ret;
    fp = fopen(file, "rb");
    if (!fp) {
        printf("bmp_load: failed to open file %s !\n", file);
        return -1;
    }

    ret = fread(&header, sizeof(header), 1, fp);
    pb->width  = header.biWidth;
    pb->height = header.biHeight;
    pb->stride = ALIGN(header.biWidth * 3, 4);
    pb->pdata  = malloc(pb->stride * pb->height);
    if (pb->pdata) {
        pdata  = (uint8_t*)pb->pdata + pb->stride * pb->height;
        for (i=0; i<pb->height; i++) {
            pdata -= pb->stride;
            ret = fread(pdata, pb->stride, 1, fp);
        }
    } else {
        printf("bmp_load: failed to allocate memory !\n");
    }

    fclose(fp);
    return pb->pdata ? 0 : -1;
}

void bmp_free(BMP *pb)
{
    if (pb->pdata) {
        free(pb->pdata);
        pb->pdata = NULL;
    }
    pb->width  = 0;
    pb->height = 0;
    pb->stride = 0;
}

float bmp_getpixel(BMP *pb, int i)
{
    uint8_t *pbyte = pb->pdata;
    int x = i % pb->width;
    int y = i / pb->width;
    int r, g, b;
    if (y >= pb->height) return 0;
    r = pbyte[x * 3 + 0 + y * pb->stride];
    g = pbyte[x * 3 + 1 + y * pb->stride];
    b = pbyte[x * 3 + 2 + y * pb->stride];
    return ((r + g + b) - 382.5) / 382.5;
}

