import pygame as pg
import numpy as np

### Parameters
s = 30      # Scaling Factor

# Origin of frame w.r.t screen
x0 = 12
y0 = 12

#Screen Dimensions
WIDTH = 20
HEIGHT = 30
screen = pg.display.set_mode((WIDTH*s, HEIGHT*s))


def tup2vec(tup):
    return np.matrix([[tup[0]], [tup[1]]])


O_fs = (x0*s, y0*s)

def screen2frame(x, v=tup2vec(O_fs)):
    """Transform screen coordinates to frame coordinates
    Coordinates are tuples
    Note: Transform only after scaling"""
    xm = tup2vec(x)
    T = np.matrix([[0, -1], [-1, 0]])  # frame rotation to screen
    ym =  T@(xm - v)
    return (ym[0,0], ym[1,0])


O_sf = screen2frame((0,0))

def frame2screen(x, v=tup2vec(O_sf)):
    """Transform frame coordinates to screen coordinates
    Coordinates are tuples
    Note: Transform only after scaling"""
    xm = tup2vec(x)
    T = np.matrix([[0, -1], [-1, 0]]) # Secren ratation to frame
    ym =  T@(xm - v)
    return (ym[0,0], ym[1,0])


# Car A Coordinates in frame
rr_a = (2, -1.952)
fr_a = (6.198, -1.952)
rl_a = (2, 0)
fl_a = (6.198, 0)
frame_coord_A = [rl_a, rr_a, fl_a, fr_a]
coord_A = [frame2screen((p[0]*s, p[1]*s)) for p in frame_coord_A]
xc_a = [p[0] for p in coord_A]
yc_a = [p[1] for p in coord_A]
left_a, top_a = frame2screen(tuple([s*p for p in fl_a]))
height_a, width_a = (max(yc_a) - min(yc_a)), (max(xc_a) - min(xc_a))

carA = pg.Rect(left_a, top_a, width_a, height_a)



# Car B Coordinates in frame
rl_b = (-10.198, 0)
rr_b = (-10.198, -1.952)
fl_b = (-6, 0)
fr_b = (-6, -1.952)
frame_coord_B = [rl_b, rr_b, fl_b, fr_b]
coord_B = [frame2screen((p[0]*s, p[1]*s)) for p in frame_coord_B]
xc_b = [p[0] for p in coord_B]
yc_b = [p[1] for p in coord_B]
left_b, top_b = frame2screen(tuple([s*p for p in fl_b]))
height_b, width_b = (max(yc_b) - min(yc_b)), (max(xc_b) - min(xc_b))

carB = pg.Rect(left_b, top_b, width_b, height_b)



# Curb
y_crb = -2.002
edge_p = frame2screen((0, y_crb*s))
yc_c = yc_a+yc_b
height_c = (max(yc_c) - min(yc_c))
width_c = width_a+width_b
top_c = min(yc_c)
left_c = edge_p[0]

Curb = pg.Rect(left_c, top_c, width_c, height_c)


class Car():
    def __init__(self, trej):
        self.x = 0      # m
        self.y = 0      # m
        self.theta = 0      # rad
        self.v = 0      # m/s
        self.phi = 0    # rad
        self.L = 2.468      # m (Wheel Base)
        self.t_trej = trej[0]
        self.trej = trej[1]

    def update_state(self, t):
        """update the state by interpolating the trajectory"""
        self.x = np.interp(t, self.t_trej, self.trej[:, 0])
        self.y = np.interp(t, self.t_trej, self.trej[:, 1])
        self.theta = np.interp(t, self.t_trej, self.trej[:, 2])
        self.v = np.interp(t, self.t_trej, self.trej[:, 3])
        self.phi = np.interp(t, self.t_trej, self.trej[:, 4])

def get_car(Car):
    """ Draw car on the screen with right coordinates"""
    lc = 0.819+2.468+0.911      # length of car in m
    wc = 1.952      # width of car in m
    d = (lc/2) - 0.819
    width = s*wc
    length = s*lc
    car_surf = pg.Surface((width, length))
    car_surf.fill((0,0,0))
    car_surf.set_colorkey((255, 0, 0))   # For transperancy of pixels
    pg.draw.rect(car_surf, (255, 255, 255), pg.Rect(0,0, width, length))
    #Wheel dimensions
    w_thick = 0.15
    w_dia = 0.8
    # rear wheels
    pg.draw.rect(car_surf, (40,0,0), pg.Rect((0.3-w_thick/2)*s, ((lc-0.819)-w_dia/2)*s,
        w_thick*s, w_dia*s))
    pg.draw.rect(car_surf, (40,0,0), pg.Rect(width-w_thick*s - (0.3-w_thick/2)*s,
        ((lc-0.819)-w_dia/2)*s, w_thick*s, w_dia*s))
    #front wheel
    w_surf = pg.Surface((w_thick*s, w_dia*s))
    w_surf.set_colorkey((255, 0, 0))
    pg.draw.rect(w_surf, (40, 0, 0), pg.Rect(0, 0, w_thick*s, w_dia*s))
    w_rot = pg.transform.rotate(w_surf, np.rad2deg(Car.phi))
    w_rot2 = w_rot.copy()
    w_rect1 = w_rot.get_rect()
    w_rect1.center = (0.3*s, 0.911*s)
    w_rect2 = w_rot2.get_rect()
    w_rect2.center = (width-(w_thick/2)*s-0.3*s, 0.911*s)
    car_surf.blit(w_rot, w_rect1)
    car_surf.blit(w_rot2, w_rect2)
    # Coordinates of the center of the car
    xc = (d*np.cos(Car.theta) + Car.x)*s
    yc = (d*np.sin(Car.theta) + Car.y)*s
    # Rotating the surface
    car_body = pg.transform.rotate(car_surf, np.rad2deg(Car.theta))
    car_rect = car_body.get_rect()
    car_rect.center = frame2screen((xc, yc))
    return car_body, car_rect




if __name__ == "__main__":
    pg.draw.rect(screen, (255, 255, 255), carA)
    pg.draw.rect(screen, (255, 255, 255), carB)
    pg.draw.rect(screen, (255, 255, 255), Curb)

    c = Car()
    c.x = 2.319
    c.y = 1.976
    c.theta = np.pi/6
    c.phi = 0
    body, rect = get_car(c)
    screen.blit(body, rect)
    pg.display.flip()
