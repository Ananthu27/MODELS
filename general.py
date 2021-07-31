### FUNCTION TO FIND DISTANCE BETWEEN TWO N DIMENTIONAL POINTS P1 AND P2
def eulidianDistance(p1,p2):
    if len(p1) and len(p1) == len(p2):
        result = [(p1[i]-p2[i])**2 for i,discard in enumerate(p1)]
        return (sum(result))**0.5
    return None