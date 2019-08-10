#!/bin/sh

set -e

gcc bitmap.c  matrix.c  samples.c  stdafx.c ffann.c -lm -Wall


