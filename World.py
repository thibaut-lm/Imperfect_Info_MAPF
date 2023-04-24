from Agent import *
from MapGenerator import *
from typing import TypedDict

import copy
import numpy as np

class World:
    
    def __init__(self, map_generator:MapGenerator, num_agents:int, broadcast_distance, view_distance):
        self.__map_generator:MapGenerator = map_generator
        
        self.__agents = {}
        self.__num_agents = num_agents
        
        self.__state_map = None
        self.__goal_map = None

        self.__view_distance = view_distance

        self.__broadcast_distance = broadcast_distance
        self.__broadcast_matrix = None
        
        self.reset()
        
    def reset(self):
        self.__state_map = self.__map_generator.generate()
        self.__agents = {id: copy.deepcopy(Agent(id)) for id in range(1, self.__num_agents + 1)}
        self.__put_agents_on_state_map()

        self.update_broadcast_matrix()
        
        self.__goal_map = np.zeros((self.__state_map.shape[0], self.__state_map.shape[1])).astype(int)
        self.__put_goals_on_goal_map()

        for agent in self.__agents.values():
            agent.init_knowledge(self.agent_world_observation(agent))
    
    def agent_world_observation(self, agent):
        agent_x, agent_y = agent.get_current_position()
        walls_map = copy.deepcopy(self.get_walls_map())
        agent_X_view = (max(0, agent_x - self.__view_distance), min(walls_map.shape[0], agent_x + self.__view_distance+1))
        agent_Y_view = (max(0, agent_y - self.__view_distance), min(walls_map.shape[1], agent_y + self.__view_distance+1))
        agent_knowledge = np.zeros(walls_map.shape).astype(int)
        
        for x in range(agent_X_view[0], agent_X_view[1]):
            for y in range(agent_Y_view[0], agent_Y_view[1]):
                agent_knowledge[x,y] = walls_map[x,y]
        
        return agent_knowledge
    
    def get_size(self):
        return self.__state_map.shape[0], self.__state_map.shape[1]
    
    def get_state_map(self):
        return self.__state_map

    def get_walls_positions(self):
        return np.argwhere(self.__state_map == -1)
    
    def get_walls_map(self):
        return (self.__state_map == -1).astype(int) 
    
    def get_agents_position_map(self):
        agent_positions_map = copy.deepcopy(self.__state_map)
        agent_positions_map[agent_positions_map < 0] = 0
        return agent_positions_map
    
    def get_goal_map(self):
        return self.__goal_map
    
    def get_agents(self):
        return self.__agents
    
    def get_agent(self, id):
        return self.__agents[id]
    
    def __free_positions(self):
        return np.argwhere(self.__state_map == 0)
    
    def clear_maps(self):
        self.__state_map[self.__state_map > 0] = 0 #remove agent from them state_map
        self.__goal_map[self.__goal_map != 0] = 0 #remove agent goals from the goal_map
        
    def update_maps(self):
        for agent_id, agent in self.__agents.items():                
            self.__state_map[agent.get_current_position()] = agent_id
            if not agent.has_reached_goal():
                self.__goal_map[agent.get_goal_position()] = agent_id
        
        self.update_broadcast_matrix()

    def update_broadcast_matrix(self):
        self.__broadcast_matrix = np.zeros((self.__num_agents, self.__num_agents))
        for a_id, agent in self.__agents.items():
            agent_x, agent_y = agent.get_current_position()
            for n_id, neighboor in self.__agents.items():
                if a_id == n_id:
                    continue
                
                neighboor_x, neighboor_y = neighboor.get_current_position()
                if abs(agent_x-neighboor_x) <= self.__broadcast_distance and abs(agent_y-neighboor_y) <= self.__broadcast_distance:
                    self.__broadcast_matrix[a_id-1, n_id-1] = 1
        
        #print(self.__broadcast_matrix)

    def get_active_agents(self):
        return {agent_id: agent for agent_id, agent in self.__agents.items() if not agent.has_reached_goal()}
            
        
    def __put_agents_on_state_map(self):
        free_positions = self.__free_positions()
        rand_indexes = np.random.choice(len(free_positions), size = len(self.__agents), replace = False)
        init_agent_positions = [tuple(free_positions[index]) for index in rand_indexes]
        
        for agent_id, agent in self.__agents.items():
            agent.move(init_agent_positions[agent_id - 1])
            self.__state_map[agent.get_current_position()] = agent.get_id()
    
    def __put_goals_on_goal_map(self):
        free_positions = self.__free_positions()
        rand_indexes = np.random.choice(len(free_positions), size = len(self.__agents), replace = False)
        init_goals_positions = [tuple(free_positions[index]) for index in rand_indexes]
        
        for agent_id, agent in self.__agents.items():
            agent.set_goal(init_goals_positions[agent_id - 1])
            self.__goal_map[agent.get_goal_position()] = agent.get_id()
    
    def simulate_joint_action(self, joint_action: list[(MovementAction, BroadcastAction)]):
        """
        take the joint action of the world active agents and simulates it on the current world

        Args:
            joint_action (list[(MovementAction:int, BroadCastAction:int)]): the joint action of the active agents to simulates on the map
        Return:
        returns the agent's status and new position after the joint action :
        - if collision is detected the agent will stay on it's current position and get a collision status
        - else the new agent position will be the the one after the action played
        """
        
        status_update = {agent_id:None for agent_id in joint_action.keys()}
        not_checked = list(joint_action.keys()) 
        wanted_positions:dict[int:tuple] = dict()
        valid_positions:dict[int:tuple] = dict()
        
        # compute wanted positions
        for current_agent_id in copy.deepcopy(not_checked):
            direction = movement_action_to_direction(joint_action[current_agent_id])
            wanted_position = tuple_plus(self.__agents[current_agent_id].get_current_position(), direction) # current position + action movement
            wanted_positions.update({current_agent_id: wanted_position})
            
        # validate stay actions
        for current_agent_id in copy.deepcopy(not_checked):
            if joint_action[current_agent_id] is MovementAction.STAY:
                status_update.update({current_agent_id: Status.STANDED_STILL})
                valid_positions.update({current_agent_id: wanted_positions[current_agent_id]})
                not_checked.remove(current_agent_id)
        
        # env collision
        for current_agent_id in copy.deepcopy(not_checked):
            if self.check_oob_and_wall_collision(wanted_positions[current_agent_id]):
                # the wanted potision is "out of band" or a wall --> STAY AT THE SAME POSITION
                status_update.update({current_agent_id: Status.COLLIDED})
                valid_positions.update({current_agent_id: self.__agents[current_agent_id].get_current_position()}) # stay on position
                wanted_positions.update({current_agent_id: self.__agents[current_agent_id].get_current_position()})
                
                not_checked.remove(current_agent_id)
        
        # swap detection
        for current_agent_id in copy.deepcopy(not_checked):
            state_of_wanted_position = self.__state_map[wanted_positions[current_agent_id]]
            if state_of_wanted_position != 0: # someone is in this position with this id
                other_agent_id = state_of_wanted_position
                if wanted_positions[other_agent_id] == self.__agents[current_agent_id].get_current_position(): # swap detection
                    if status_update[current_agent_id] is None:
                        status_update.update({current_agent_id: Status.COLLIDED})
                        valid_positions.update({current_agent_id: self.__agents[current_agent_id].get_current_position()}) # stay on position
                        wanted_positions.update({current_agent_id: self.__agents[current_agent_id].get_current_position()})
                        
                        not_checked.remove(current_agent_id)
                    
                    if status_update[other_agent_id] is None:
                        status_update.update({other_agent_id: Status.COLLIDED})
                        valid_positions.update({other_agent_id: self.__agents[other_agent_id].get_current_position()}) # stay on position
                        wanted_positions.update({other_agent_id: self.__agents[other_agent_id].get_current_position()})
                        
                        not_checked.remove(other_agent_id)
        
        # cellwise collision
        for current_agent_id in copy.deepcopy(not_checked):
            other_agent_wanted_positions = copy.deepcopy(wanted_positions)
            other_agent_wanted_positions.pop(current_agent_id)
            
            if wanted_positions[current_agent_id] in valid_positions.values(): #someone already validated this position
                status_update.update({current_agent_id: Status.COLLIDED})
                valid_positions.update({current_agent_id: self.__agents[current_agent_id].get_current_position()}) # stay on position
                wanted_positions.update({current_agent_id: self.__agents[current_agent_id].get_current_position()})
                
                not_checked.remove(current_agent_id)
            
            elif wanted_positions[current_agent_id] in other_agent_wanted_positions.values(): # multiple collision detection on same point
                other_agents = [agent_id for agent_id, wanted_position in other_agent_wanted_positions.items() if wanted_position == wanted_positions[current_agent_id]]
                
                if current_agent_id < min(other_agents): # valid move for lowest agent id
                    if not self.__agents[current_agent_id].has_reached_goal() and wanted_positions[current_agent_id] == self.__agents[current_agent_id].get_goal_position():
                        status_update.update({current_agent_id: Status.MOVED_TO_GOAL})
                    else:
                        status_update.update({current_agent_id: Status.MOVED})
                    
                    valid_positions[current_agent_id] = wanted_positions[current_agent_id] # move to wanted pos
                    not_checked.remove(current_agent_id)
                
                else: # invalid move for other agents
                    status_update.update({current_agent_id: Status.COLLIDED})
                    valid_positions.update({current_agent_id: self.__agents[current_agent_id].get_current_position()}) # stay on position
                    wanted_positions.update({current_agent_id: self.__agents[current_agent_id].get_current_position()})
                
                    not_checked.remove(current_agent_id)
                    
        # all the rest actions are valid
        for current_agent_id in copy.deepcopy(not_checked):
            if not self.__agents[current_agent_id].has_reached_goal() and wanted_positions[current_agent_id] == self.__agents[current_agent_id].get_goal_position(): #check for first time goal reaching
                status_update.update({current_agent_id: Status.MOVED_TO_GOAL})
            else:
                status_update.update({current_agent_id: Status.MOVED})

            valid_positions[current_agent_id] = wanted_positions[current_agent_id] # move to wanted pos
            not_checked.remove(current_agent_id)
            
        assert not not_checked
        
        return status_update, valid_positions
        
    def check_oob_and_wall_collision(self, wanted_position:tuple) -> bool:
        # wall
        if self.__state_map[wanted_position] == -1:
            return True
        # oob
        if wanted_position[0] < 0 or wanted_position[0] > self.__state_map.shape[0] \
        or wanted_position[1] < 0 or wanted_position[1] > self.__state_map.shape[1] :
            return True
        # no wall neither oob
        return False
    
    def compute_broadcast(self, broadcast_actions:dict[int:BroadcastAction]):
        for a_id in self.__agents.keys():
            for n_id in self.__agents.keys():
                if a_id == n_id : continue
                
                if self.__broadcast_matrix[a_id-1,n_id-1] == 1 and broadcast_actions[n_id] == BroadcastAction.ON:
                    #if neighboor is broadcasting i update my knowledge
                    self.__agents[a_id].update_knowledge(self.__agents[n_id].get_world_knowledge())
