from operator import sub, add
from enum import Enum
import numpy as np
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

class Status(Enum):
    STANDED_STILL = 0
    MOVED = 1
    MOVED_TO_GOAL = 2
    COLLIDED = 3

class BroadcastAction(Enum):
    OFF = 0
    ON = 1
        
class MovementAction(Enum):
    STAY = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    UP = 4
    
#lines = x, colomns = y
DIRECTION = {
    MovementAction.STAY  : (0,0),
    MovementAction.RIGHT : (0,1),
    MovementAction.DOWN  : (1,0),
    MovementAction.LEFT  : (0,-1),
    MovementAction.UP    : (-1,0)
}

def movement_action_to_direction(move_action:MovementAction) -> tuple:
    return DIRECTION[move_action]

def direction_to_movement_action(direction:tuple) -> MovementAction:
    return [move_action for move_action, dir in DIRECTION.items() if dir == direction][0]

def tuple_plus(a,b):
    return tuple(map(add,a,b))

def tuple_minus(a,b):
    return tuple(map(sub,a,b))

def heap(ls, max_length):
    while len(ls) > max_length:
        ls.pop()
    return ls

class Agent:
    
    def __init__(self, id):
        self.__id:int = id
        self.__current_position:tuple = None
        self.__goal_position:tuple = None
        self.__status: Status = None
        self.__reached_goal = False
        self.__world_knowledge = None

    def init_knowledge(self, world_knowledge):
        self.__world_knowledge = world_knowledge
        
    def update_knowledge(self, new_observation):
        self.__world_knowledge = np.maximum(self.__world_knowledge, new_observation)
    
    def get_world_knowledge(self):
        return self.__world_knowledge

    def get_id(self) -> id:
        return self.__id
    
    def get_current_position(self) -> tuple:
        return self.__current_position
    
    def get_goal_position(self) -> tuple:
        return self.__goal_position
    
    def get_status(self) -> Status:
        return self.__status
        
    def has_reached_goal(self):
        return self.__reached_goal
    
    def move(self, new_position:tuple, new_status:Status = None):
        self.__current_position = new_position
        self.__status = new_status
        
        if self.__current_position == self.__goal_position:
            self.__reached_goal = True
        
    def set_goal(self, new_goal_position: tuple):
        self.__goal_position = new_goal_position
        self.__reached_goal = False

    def path_to_goal(self, finder):
        grid = Grid(matrix=1-self.__world_knowledge)
        start = grid.node(self.get_current_position()[1], self.get_current_position()[0])
        end = grid.node(self.get_goal_position()[1], self.get_goal_position()[0])
        path, runs = finder.find_path(start, end, grid)
        return path