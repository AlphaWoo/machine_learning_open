# -*- coding: utf-8 -*-
for x in range(1,5):
    for y in range(1,5):
        for z in range(1,5):
                if (x!=y) and (y!=z) and (z!=x):
                        print("%d%d%d" % (x, y, z))