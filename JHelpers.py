''' Helper file for python functions '''
import numpy
''' Distance based on x1,y1,x2,y2 inputs '''
def distance(x1,y1,x2,y2):
    a = numpy.array([x1,y1])
    b = numpy.array([x2,y2])
    return numpy.linalg.norm(a-b)

def npdistance(a,b):
    return numpy.linalg.norm(a-b)