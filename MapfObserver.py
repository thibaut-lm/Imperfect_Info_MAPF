import numpy as np

from World import *


class MapfObserver:
    
    def __init__(self, view_distance, broadcast_distance):
        self.__world:World = None
        self.__view_distance = view_distance
        self.__broadcast_distance = broadcast_distance
    
    def set_world(self, world):
        self.__world = world
    
    def get_view_distance(self):
        return self.__view_distance
        
    def get_broadcast_distance(self):
        return self.__broadcast_distance
    
    def observe(self):        
        
        agents_map = self.__world.get_agents_position_map()
        goals_map = self.__world.get_goal_map()
        walls_map = self.__world.get_walls_map()
        return np.array([agents_map, goals_map, walls_map], dtype=np.uint8)
    
    def observe_agent(self, agent_id):
        agents_map = self.__world.get_agents_position_map()
        goals_map = self.__world.get_goal_map()
        walls_map = self.__world.get_walls_map()
        return np.array([agents_map, goals_map, walls_map], dtype=np.uint8)