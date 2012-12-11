/*
  convo2d: Spatial convolution filter for VapourSynth

  Copyright (C) 2012  Oka Motofumi

  Author: Oka Motofumi (chikuzen.mo at gmail dot com)

  This program is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with the author; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <stdint.h>
#include <math.h>
#include "VapourSynth.h"

#ifdef _MSC_VER
#pragma warning(disable:4996)
#define snprintf _snprintf
#endif

#define CONVO2D_VERSION "0.1.1"

typedef struct convo2d_handle convo2d_t;

struct convo2d_handle {
    VSNodeRef *node;
    const VSVideoInfo *vi;
    int m[25];
    uint16_t max;
    void (VS_CC *proc_convolution)(convo2d_t *ch, int plane,
                                   const VSFrameRef *src, VSFrameRef *dst,
                                   const VSAPI *vsapi);
    double div;
    double bias;
    int planes[3];
};

static uint16_t clamp(double val, uint16_t max)
{
    if (val < 0) {
        return 0;
    }
    if (val > max) {
        return max;
    }
    return (uint16_t)val;
}


#define CONVOLUTION_3x3(X0, X1, X2) (\
    (*(r0 + (X0)) * m00) + (*(r0 + (X1)) * m01) + (*(r0 + (X2)) * m02) + \
    (*(r1 + (X0)) * m10) + (*(r1 + (X1)) * m11) + (*(r1 + (X2)) * m12) + \
    (*(r2 + (X0)) * m20) + (*(r2 + (X1)) * m21) + (*(r2 + (X2)) * m22))

#define WRITE_DSTP_3x3(X0, X1, X2, T) \
{\
    int64_t value = CONVOLUTION_3x3(X0, X1, X2);\
    dstp[(X1)] = (T)clamp(value / ch->div + ch->bias, ch->max);\
}

#define PROC_3x3(T) \
{\
    WRITE_DSTP_3x3(0, 0, 1, T); \
    for (int x = 1; x < w; x++) { \
        int xl = x - 1, xr = x + 1; \
        WRITE_DSTP_3x3(xl, x, xr, T); \
    } \
    WRITE_DSTP_3x3(w - 1, w, w, T); \
    r1 += stride; \
    dstp += stride; \
}

static void VS_CC
proc_3x3_8bit(convo2d_t *ch, int plane, const VSFrameRef *src, VSFrameRef *dst,
              const VSAPI *vsapi)
{
    int w = vsapi->getFrameWidth(src, plane) - 1;
    int h = vsapi->getFrameHeight(src, plane) - 1;
    if (w < 2 || h < 2) {
        return;
    }
    int stride = vsapi->getStride(src, plane);
    uint8_t *dstp = vsapi->getWritePtr(dst, plane);
    int m00 = ch->m[0], m01 = ch->m[1], m02 = ch->m[2],
        m10 = ch->m[3], m11 = ch->m[4], m12 = ch->m[5],
        m20 = ch->m[6], m21 = ch->m[7], m22 = ch->m[8];

    /* top */
    const uint8_t *r0 = vsapi->getReadPtr(src, plane);
    const uint8_t *r1 = r0;
    const uint8_t *r2 = r1 + stride;
    PROC_3x3(uint8_t);

    /* center */
    for (int y = 1; y < h; y++) {
        r0 = r1 - stride;
        r2 = r1 + stride;
        PROC_3x3(uint8_t);
    }

    /* bottom */
    r0 = r1 - stride;
    PROC_3x3(uint8_t);
}


static void VS_CC
proc_3x3_16bit(convo2d_t *ch, int plane, const VSFrameRef *src, VSFrameRef *dst,
               const VSAPI *vsapi)
{
    int w = vsapi->getFrameWidth(src, plane) - 1;
    int h = vsapi->getFrameHeight(src, plane) - 1;
    if (w < 2 || h < 2) {
        return;
    }
    int stride = vsapi->getStride(src, plane) / 2;
    uint16_t *dstp = (uint16_t *)vsapi->getWritePtr(dst, plane);
    int m00 = ch->m[0], m01 = ch->m[1], m02 = ch->m[2],
        m10 = ch->m[3], m11 = ch->m[4], m12 = ch->m[5],
        m20 = ch->m[6], m21 = ch->m[7], m22 = ch->m[8];

    /* top */
    const uint16_t *r0 = (uint16_t *)vsapi->getReadPtr(src, plane);
    const uint16_t *r1 = r0;
    const uint16_t *r2 = r1 + stride;
    PROC_3x3(uint16_t);

    /* center */
    for (int y = 1; y < h; y++) {
        r0 = r1 - stride;
        r2 = r1 + stride;
        PROC_3x3(uint16_t);
    }

    /* bottom */
    r0 = r1 - stride;
    PROC_3x3(uint16_t);
}
#undef PROC_3x3
#undef WRITE_DSTP_3x3
#undef CONVOLUTION_3x3


#define CONVOLUTION_5x5(X0, X1, X2, X3, X4) (\
    (*(r0 + (X0)) * m00) + (*(r0 + (X1)) * m01) + (*(r0 + (X2)) * m02) + (*(r0 + (X3)) * m03) + (*(r0 + (X4)) * m04) +\
    (*(r1 + (X0)) * m10) + (*(r1 + (X1)) * m11) + (*(r1 + (X2)) * m12) + (*(r1 + (X3)) * m13) + (*(r1 + (X4)) * m14) +\
    (*(r2 + (X0)) * m20) + (*(r2 + (X1)) * m21) + (*(r2 + (X2)) * m22) + (*(r2 + (X3)) * m23) + (*(r2 + (X4)) * m24) +\
    (*(r3 + (X0)) * m30) + (*(r3 + (X1)) * m31) + (*(r3 + (X2)) * m32) + (*(r3 + (X3)) * m33) + (*(r3 + (X4)) * m34) +\
    (*(r4 + (X0)) * m40) + (*(r4 + (X1)) * m41) + (*(r4 + (X2)) * m42) + (*(r4 + (X3)) * m43) + (*(r4 + (X4)) * m44))

#define WRITE_DSTP_5x5(X0, X1, X2, X3, X4, T) \
{\
    int64_t value = CONVOLUTION_5x5(X0, X1, X2, X3, X4); \
    dstp[(X2)] = (T)clamp(value / ch->div + ch->bias, ch->max); \
}

#define PROC_5x5(T) \
{\
    WRITE_DSTP_5x5(0, 0, 0, 1, 2, T); \
    WRITE_DSTP_5x5(0, 0, 1, 2, 3, T); \
    for (int x = 2; x < w; x++) { \
        int x0 = x - 2, x1 = x - 1, x3 = x + 1, x4 = x + 2; \
        WRITE_DSTP_5x5(x0, x1, x, x3, x4, T); \
    } \
    WRITE_DSTP_5x5(w - 2, w - 1, w, w + 1, w + 1, T); \
    WRITE_DSTP_5x5(w - 1, w, w + 1, w + 1, w + 1, T); \
    r2 += stride; \
    dstp += stride; \
}

static void VS_CC
proc_5x5_8bit(convo2d_t *ch, int plane, const VSFrameRef *src, VSFrameRef *dst,
              const VSAPI *vsapi)
{
    int w = vsapi->getFrameWidth(src, plane) - 2;
    int h = vsapi->getFrameHeight(src, plane) - 2;
    if (w < 3 || h < 3) {
        return;
    }
    int stride = vsapi->getStride(src, plane);
    uint8_t *dstp = vsapi->getWritePtr(dst, plane);
    int m00 = ch->m[ 0], m01 = ch->m[ 1], m02 = ch->m[ 2], m03 = ch->m[ 3], m04 = ch->m[ 4],
        m10 = ch->m[ 5], m11 = ch->m[ 6], m12 = ch->m[ 7], m13 = ch->m[ 8], m14 = ch->m[ 9],
        m20 = ch->m[10], m21 = ch->m[11], m22 = ch->m[12], m23 = ch->m[13], m24 = ch->m[14],
        m30 = ch->m[15], m31 = ch->m[16], m32 = ch->m[17], m33 = ch->m[18], m34 = ch->m[19],
        m40 = ch->m[20], m41 = ch->m[21], m42 = ch->m[22], m43 = ch->m[23], m44 = ch->m[24];

    /* top */
    const uint8_t *r0 = vsapi->getReadPtr(src, plane);
    const uint8_t *r1 = r0;
    const uint8_t *r2 = r0;
    const uint8_t *r3 = r2 + stride;
    const uint8_t *r4 = r3 + stride;
    PROC_5x5(uint8_t);

    /* top + 1 */
    r3 += stride;
    r4 += stride;
    PROC_5x5(uint8_t);

    /* center */
    for (int y = 2; y < h; y++) {
        r1 = r2 - stride;
        r0 = r1 - stride;
        r3 = r2 + stride;
        r4 = r3 + stride;
        PROC_5x5(uint8_t);
    }

    /* bottom - 1 */
    r1 = r2 - stride;
    r0 = r1 - stride;
    r3 = r2 + stride;
    PROC_5x5(uint8_t);

    /* bottom */
    r1 = r2 - stride;
    r0 = r1 - stride;
    PROC_5x5(uint8_t);
}


static void VS_CC
proc_5x5_16bit(convo2d_t *ch, int plane, const VSFrameRef *src, VSFrameRef *dst,
               const VSAPI *vsapi)
{
    int w = vsapi->getFrameWidth(src, plane) - 2;
    int h = vsapi->getFrameHeight(src, plane) - 2;
    if (w < 3 || h < 3) {
        return;
    }
    int stride = vsapi->getStride(src, plane) / 2;
    uint16_t *dstp = (uint16_t *)vsapi->getWritePtr(dst, plane);
    int m00 = ch->m[ 0], m01 = ch->m[ 1], m02 = ch->m[ 2], m03 = ch->m[ 3], m04 = ch->m[ 4],
        m10 = ch->m[ 5], m11 = ch->m[ 6], m12 = ch->m[ 7], m13 = ch->m[ 8], m14 = ch->m[ 9],
        m20 = ch->m[10], m21 = ch->m[11], m22 = ch->m[12], m23 = ch->m[13], m24 = ch->m[14],
        m30 = ch->m[15], m31 = ch->m[16], m32 = ch->m[17], m33 = ch->m[18], m34 = ch->m[19],
        m40 = ch->m[20], m41 = ch->m[21], m42 = ch->m[22], m43 = ch->m[23], m44 = ch->m[24];

    /* top */
    const uint16_t *r0 = (uint16_t *)vsapi->getReadPtr(src, plane);
    const uint16_t *r1 = r0;
    const uint16_t *r2 = r0;
    const uint16_t *r3 = r2 + stride;
    const uint16_t *r4 = r3 + stride;
    PROC_5x5(uint16_t);

    /* top + 1 */
    r3 += stride;
    r4 += stride;
    PROC_5x5(uint16_t);

    /* center */
    for (int y = 2; y < h; y++) {
        r1 = r2 - stride;
        r0 = r1 - stride;
        r3 = r2 + stride;
        r4 = r3 + stride;
        PROC_5x5(uint16_t);
    }

    /* bottom - 1 */
    r1 = r2 - stride;
    r0 = r1 - stride;
    r3 = r2 + stride;
    PROC_5x5(uint16_t);

    /* bottom */
    r1 = r2 - stride;
    r0 = r1 - stride;
    PROC_5x5(uint16_t);
}
#undef PROC_5x5
#undef WRITE_DSTP_5x5
#undef CONVOLUTION_5x5


static const VSFrameRef * VS_CC
convo2d_get_frame(int n, int activation_reason, void **instance_data,
                  void **frame_data, VSFrameContext *frame_ctx,
                  VSCore *core, const VSAPI *vsapi)
{
    convo2d_t *ch = (convo2d_t *)*instance_data;

    if (activation_reason == arInitial) {
        vsapi->requestFrameFilter(n, ch->node, frame_ctx);
        return NULL;
    }

    if (activation_reason != arAllFramesReady) {
        return NULL;
    }

    const VSFrameRef *src = vsapi->getFrameFilter(n, ch->node, frame_ctx);
    const VSFormat *fi = vsapi->getFrameFormat(src);
    const int pl[] = {0, 1, 2};
    const VSFrameRef *fr[] = {ch->planes[0] ? NULL : src,
                              ch->planes[1] ? NULL : src,
                              ch->planes[2] ? NULL : src};
    VSFrameRef *dst = vsapi->newVideoFrame2(fi, vsapi->getFrameWidth(src, 0),
                                            vsapi->getFrameHeight(src, 0),
                                            fr, pl, src, core);

    for (int plane = 0; plane < fi->numPlanes; plane++) {
        if (fr[plane]) {
            continue;
        }

        ch->proc_convolution(ch, plane, src, dst, vsapi);
    }

    return dst;
}


static void VS_CC
init_convo2d(VSMap *in, VSMap *out, void **instance_data, VSNode *node,
             VSCore *core, const VSAPI *vsapi)
{
    convo2d_t *ch = (convo2d_t *)*instance_data;
    vsapi->setVideoInfo(ch->vi, 1, node);
    vsapi->clearMap(in);
}


static void VS_CC close_handler(convo2d_t *ch, const VSAPI *vsapi)
{
    if (!ch) {
        return;
    }
    if (ch->node) {
        vsapi->freeNode(ch->node);
        ch->node = NULL;
    }
    free(ch);
    ch = NULL;
}


static void VS_CC
close_convo2d(void *instance_data, VSCore *core, const VSAPI *vsapi)
{
    convo2d_t *ch = (convo2d_t *)instance_data;
    close_handler(ch, vsapi);
}


#define RET_IF_ERROR(cond, ...) \
{\
    if (cond) {\
        close_handler(ch, vsapi);\
        snprintf(msg, 240, __VA_ARGS__);\
        vsapi->setError(out, msg_buff);\
        return;\
    }\
}

static void VS_CC
create_convo2d(const VSMap *in, VSMap *out, void *user_data, VSCore *core,
               const VSAPI *vsapi)
{
    char msg_buff[256] = "convolution: ";
    char *msg = msg_buff + strlen(msg_buff);

    convo2d_t *ch = (convo2d_t *)calloc(sizeof(convo2d_t), 1);
    RET_IF_ERROR(!ch, "failed to allocate handler");

    ch->node = vsapi->propGetNode(in, "clip", 0, 0);
    ch->vi = vsapi->getVideoInfo(ch->node);

    int num = vsapi->propNumElements(in, "planes");
    if (num < 1) {
        for (int i = 0; i < ch->vi->format->numPlanes; ch->planes[i++] = 1);
    } else {
        for (int i = 0; i < num; i++) {
            int p = (int)vsapi->propGetInt(in, "planes", i, NULL);
            RET_IF_ERROR(p < 0 || p >= ch->vi->format->numPlanes,
                         "planes index out of range");
            ch->planes[p] = 1;
        }
    }

    num = vsapi->propNumElements(in, "matrix");
    RET_IF_ERROR(num > 0 && num != 9 && num != 25, "invalid matrix");
    if (num < 0 || num == 9) {
        ch->proc_convolution = proc_3x3_8bit;
        if (ch->vi->format->bytesPerSample == 2) {
            ch->proc_convolution = proc_3x3_16bit;
        }
    } else {
        ch->proc_convolution = proc_5x5_8bit;
        if (ch->vi->format->bytesPerSample == 2) {
            ch->proc_convolution = proc_5x5_16bit;
        }
    }
    ch->m[4] = 1;
    for (int i = 0; i < num; i++) {
        int element = (int)vsapi->propGetInt(in, "matrix", i, NULL);
        ch->m[i] = element;
        ch->div += element;
    }
    if (ch->div == 0.0) {
        ch->div = 1.0;
    }

    int err;
    ch->bias = vsapi->propGetFloat(in, "bias", 0, &err);
    if (err) {
        ch->bias = 0.0;
    }
    ch->bias += 0.5;

    double div = vsapi->propGetFloat(in, "divisor", 0, &err);
    if (!err && div != 0.0) {
        ch->div = div;
    }

    ch->max = (uint16_t)(pow(2.0, ch->vi->format->bitsPerSample) - 1);

    vsapi->createFilter(in, out, "Convolution", init_convo2d,
                        convo2d_get_frame, close_convo2d, fmParallel,
                        0, ch, core);
}
#undef RET_IF_ERROR


VS_EXTERNAL_API(void) VapourSynthPluginInit(
    VSConfigPlugin f_config, VSRegisterFunction f_register, VSPlugin *plugin)
{
    f_config("chikuzen.does.not.have.his.own.domain.convo2d", "convo2d",
             "Spatial convolution filter for VapourSynth v" CONVO2D_VERSION,
             VAPOURSYNTH_API_VERSION, 1, plugin);
    f_register("Convolution",
               "clip:clip;matrix:int[]:opt;bias:float:opt;divisor:float:opt;"
               "planes:int[]:opt;",
               create_convo2d, NULL, plugin);
}
