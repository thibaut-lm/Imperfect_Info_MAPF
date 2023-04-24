import numpy as np

class MapGenerator:
    
    def __init__(self, env_size, wall_components, obstacle_density, go_straight=0.8):
        
        self.__env_size = env_size //2 * 2 + 3
        self.__wall_components = wall_components
        self.__go_straight = go_straight
        self.__obstacle_density = obstacle_density
    
    def get_size(self):
        return (self.__env_size, self.__env_size)

    def __maze(self, h, w, total_density=0):
        shape = (h, w)
        density = int(shape[0] * shape[1] * total_density // self.__wall_components) if self.__wall_components != 0 else 0

        M = np.zeros(shape, dtype='int')
        M[0,:] = M[-1, :] = M[:,0] = M[:,-1] = 1

        for i in range(density):
            x,y = np.random.randint(0, shape[0]//2)*2, np.random.randint(0,shape[1]//2)*2
            M[x,y] = 1
            last_dir = 0
            for j in range(self.__wall_components):
                neighbours = []
                if x > 1:             neighbours.append((y, x - 2))
                if x < shape[1] - 2:  neighbours.append((y, x + 2))
                if y > 1:             neighbours.append((y - 2, x))
                if y < shape[0] - 2:  neighbours.append((y + 2, x))
                if len(neighbours):
                    if last_dir == 0:
                        y_, x_ = neighbours[np.random.randint(0, len(neighbours))]
                        if M[y_, x_] == 0:
                            last_dir = (y_ - y, x_ - x)
                            M[y_, x_] = 1
                            M[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                            x, y = x_, y_
                    else:
                        index_F = -1
                        index_B = -1
                        diff = []
                        for k in range(len(neighbours)):
                            diff.append((neighbours[k][0] - y, neighbours[k][1] - x))
                            if diff[k] == last_dir:
                                index_F = k
                            elif diff[k][0] + last_dir[0] == 0 and diff[k][1] + last_dir[1] == 0:
                                index_B = k
                        assert (index_B >= 0)
                        if (index_F + 1):
                            p = (1 - self.__go_straight) * np.ones(len(neighbours)) / (len(neighbours) - 2)
                            p[index_B] = 0
                            p[index_F] = self.__go_straight
                            # assert(p.sum() == 1)
                        else:
                            if len(neighbours) == 1:
                                p = 1
                            else:
                                p = np.ones(len(neighbours)) / (len(neighbours) - 1)
                                p[index_B] = 0
                            assert (p.sum() == 1)

                        I = np.random.choice(range(len(neighbours)), p=p)
                        (y_, x_) = neighbours[I]
                        if M[y_, x_] == 0:
                            last_dir = (y_ - y, x_ - x)
                            M[y_, x_] = 1
                            M[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                            x, y = x_, y_
        return M
    
    def __validate(self):
        return True
    
    def generate(self):
        validated = False
        while not validated:
            world = - self.__maze(int(self.__env_size), int(self.__env_size), total_density=self.__obstacle_density).astype(int)
            world = np.array(world)
            validated = self.__validate()

        return world