from car_sim import *
import pickle

class Game:
    def __init__(self):
        pg.init()
        pg.display.set_caption("Non-holonomic Car")
        self.screen = screen
        self.clock = pg.time.Clock()
        self.ticks = 60
        self.exit = False
        self.car = Car()
        self.car.beta = 1
        Ix = np.arange(2.319, 2.819, 0.001)
        Iy = np.arange(1.726, 1.976, 0.001)
        self.car.x = Ix[np.random.randint(0, len(Ix))]
        self.car.y = Iy[np.random.randint(0, len(Iy))]
        self.car.theta = 0
        self.x_hist = []
        self.y_hist = []
        self.theta_hist = []

    def run(self):
        t = 0
        while not self.exit:
            dt = self.clock.get_time()/1000

            # Event queue
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.exit = True

            self.car.update_state(t)
            car_body, car_rect = get_car(self.car)
            if (car_rect.colliderect(Curb)):
                self.car.revert_state()
                car_body, car_rect = get_car(self.car)

            self.x_hist.append(self.car.x)
            self.y_hist.append(self.car.y)
            self.theta_hist.append(self.car.theta)
            pickle.dump((self.car.v_RCP, self.car.phi_RCP), open("cin.bin",
            "wb"))

            # Drawing
            self.screen.fill((0, 0, 0))
            pg.draw.rect(screen, (255, 255, 255), carA)
            pg.draw.rect(screen, (255, 255, 255), carB)
            pg.draw.rect(screen, (255, 255, 255), Curb)
            self.screen.blit(car_body, car_rect)

            pg.display.flip()
            t = t+dt
            self.clock.tick(self.ticks)
        pg.quit()


if __name__ == "__main__":
    import pickle
    game = Game()
    game.run()
    with open("parking_data", 'wb') as f:
        pickle.dump([game.x_hist, game.y_hist, game.theta_hist], f)
