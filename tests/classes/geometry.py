import random
import numpy as np



class Point():
    def __init__(self, x, y, idx = -1, connected = False):
        self.idx = idx
        self.v = np.array([x,y])
        self.connected = connected

    @property
    def x(self):
        return self.v[0]

    @property
    def y(self):
        return self.v[1]

    def get_direction(self, other: "Point"):
        return other.v - self.v

    def __repr__(self):
        return f"x = {self.v[0]}, y = {self.v[1]}, idx = {self.idx}"
     
    def __eq__(self, other):
        return self.idx == other.idx and np.array_equal(self.v, other.v)



class Segment():
    def __init__(self, p1: Point, p2: Point):
        self.p1: Point = p1
        self.p2: Point = p2

    def __repr__(self):
        return f"Segment linking: \n p1 = {self.p1},\n p2 = {self.p2}"

class Points():
    def __init__(self):
        self.n_points = 0
        self.l_points : list[Point] = []
        self.dist_dict = {}
    
    @staticmethod
    def calculate_distance(p1, p2):
        return np.sqrt(np.sum((p1.v - p2.v)**2))
    
    def get_distance(self, p1, p2):
        return self.dist_dict[p1.idx, p2.idx]

    def add_point(self, x, y):
        new_point = Point(x, y, self.n_points)
        self.l_points.append(new_point)
        self.n_points += 1
        for point in self.l_points:
            d = self.calculate_distance(new_point, point)
            self.dist_dict[(point.idx, new_point.idx)] = d
            self.dist_dict[(new_point.idx, point.idx)] = d
        return new_point
    
    def pick_unconnected_point(self):
        l = []
        for i in range(self.n_points):
            if not self.l_points[i].connected:
                l.append(i)
        if len(l) > 1:
            return self.l_points[random.choice(l)]
        return None

    def pick_point(self, exception_points = []):
        point = random.choice(self.l_points)
        while point in exception_points:
            point = random.choice(self.l_points)
        return point

    def pick_points(self, n):
        return random.sample(self.l_points, k = n)

    def pick_point_dist(self, point_ref: Point, function):
        candidats = [p for p in self.l_points if p.idx != point_ref.idx]
        poids = []
        for p in candidats:
            dist = self.get_distance(point_ref, p)
            poids.append(function(dist))
        return random.choices(candidats, weights=poids, k=1)[0]

class Segment():
    def __init__(self, p1: Point, p2: Point):
        self.p1: Point = p1
        self.p2: Point = p2

    def __repr__(self):
        return f"p1 = {self.p1},\n p2 = {self.p2},\n idx = {self.idx}"
    
class Segments():
    def __init__(self):
        self.l_segments : list[Segment] = [] 
        self.n_segments = 0
    
    @staticmethod
    def intersection(s1:Segment, s2:Segment):
        x1, y1 = s1.p1.v
        x2, y2 = s1.p2.v
        x3, y3 = s2.p1.v
        x4, y4 = s2.p2.v

        h = s1 if y1 == y2 else s2 if y3 == y4 else None
        v = s2 if h == s1 else s1 if h == s2 else None

        if h and v:
            x_coords = [h.p1.v[0], h.p2.v[0]]
            y_coords = [v.p1.v[1], v.p2.v[1]]
            
            x_start, x_end = min(x_coords), max(x_coords)
            y_start, y_end = min(y_coords), max(y_coords)
            
            y_val = h.p1.v[1]
            x_val = v.p1.v[0]

            if x_start < x_val < x_end and y_start < y_val < y_end:
                return (x_val, y_val)
        return False

    def add_segment(self, p1: Point, p2: Point, points: Points):
            if np.array_equal(p1.v, p2.v):
                return

            new_segment = Segment(p1, p2)
            inter = None
            segment_collision = None
            
            for segment in self.l_segments:
                res = self.intersection(new_segment, segment)
                if res:
                    inter = res
                    segment_collision = segment
                    break

            if inter:
                x, y = inter
                p3 = points.add_point(x, y)
                p3.connected = True
                
                self.l_segments.remove(segment_collision)
                
                self.add_segment(segment_collision.p1, p3, points)
                self.add_segment(segment_collision.p2, p3, points)
                
                self.add_segment(p1, p3, points)
                self.add_segment(p2, p3, points)
            else:
                self.n_segments += 1
                p1.connected = True
                p2.connected = True
                self.l_segments.append(new_segment)
