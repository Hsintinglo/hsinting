import shapely

from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.affinity import scale
import numpy as np


def vec_on_circle(r, p1, p2):
    p = Point(0,0)
    c = p.buffer(r).boundary
    l = LineString([p1, p2])
    l2 = scale(l, 10000, 10000)
    i = c.intersection(l2)
    ics = [i.geoms[0].coords[0], \
            i.geoms[1].coords[0]]
    dist2p1 = [Point(ics[0]).distance(Point(p1)), 
            Point(ics[1]).distance(Point(p1))]
    
    min_ic2p1 = np.argmin(dist2p1)
    ic_real = ics[min_ic2p1]
    return(ic_real)