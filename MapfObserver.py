import numpy as np

from World import *
from sklearn import preprocessing


class MapfObserver:
    
    def __init__(self, view_distance, communication_distance):
        self.__world:World = None
        self.__view_distance = view_distance
        self.__communication_distance = communication_distance
        self.__finder = None
        self.num_future_steps = 3
    
    def set_world(self, world):
        self.__world = world
        
    def set_finder(self, finder):
        self.__finder = finder
    
    def get_view_distance(self):
        return self.__view_distance
        
    def get_communication_distance(self):
        return self.__communication_distance
    
    def observe(self):        
        
        observations = {}
        for agent_id, agent in self.__world.get_agents().items():
            observations.update({agent_id:self.observe_agent(agent_id)})
        
        return observations
    
    def observe_agent(self, agent_id):
        agent = self.__world.get_agent(agent_id)
        agent_x, agent_y = agent.get_current_position()
        top_left = (agent_x-self.__view_distance, agent_y-self.__view_distance)
        
        observation_size = (2*self.__view_distance) + 1
        observation_shape = ((2*self.__view_distance) + 1, (2*self.__view_distance) + 1)
        
        other_agent_map = np.zeros(observation_shape, dtype=np.uint16)
        wall_map = np.zeros(observation_shape, dtype=np.uint16)
        astar_map = np.zeros([self.num_future_steps, observation_size, observation_size], dtype=np.uint16)
        pathlength_map = np.zeros(observation_shape, dtype=np.uint16)
        
        visible_agents = []
        
        for i in range(top_left[0], top_left[0] + observation_size):
            for j in range(top_left[1], top_left[1]+ observation_size):
                obs_i, obs_j = i-top_left[0], j-top_left[1]
                
                if i >= self.__world.get_state_map().shape[0] or i <0 or j >= self.__world.get_state_map().shape[1] or j < 0:
                    #oob observation -> wall
                    wall_map[obs_i, obs_j] = 1
                    continue
                
                if not self.__world.in_view(agent_x, agent_y, i, j):
                    # Check world knowledge
                    if agent.get_world_knowledge()[i][j] == 1:
                          wall_map[obs_i, obs_j] = 1
                    
                    continue
                
                #cell[i,j] in view
                
                #start with state_map observation -> wall & other_agent map
                cell_state_value = self.__world.get_state_map()[i,j]
                
                if cell_state_value == -1:
                    #wall
                    wall_map[obs_i, obs_j] = 1
                
                if cell_state_value > 0:
                    #agent
                    if cell_state_value != agent_id:
                        other_agent_map[obs_i, obs_j] = 1 #an agent is at this position
                        visible_agents.append(cell_state_value)
                
        #continue with goal_map observation -> a_star_map
        for obs_agent_id in visible_agents:
            #compute astar_path with current agent world knowledge limited to num_future_step
            obs_agent = self.__world.get_agent(obs_agent_id)
            path = agent.path_to_goal_on_self_knowledge(self.__finder,start_position=obs_agent.get_current_position(), end_position=obs_agent.get_goal_position())
            path = path[:self.num_future_steps] #num_future_steps predicted
            if path == []: path.append(obs_agent.get_current_position())
            while len(path) < self.num_future_steps: #agent will be predicted to stay at it's goal if it's on it
                path.append(path[-1])
            
            for i, step in enumerate(path):
                #! step coordinates are not in the right order
                step_x_in_agent_view = step[1] - agent_x + self.__view_distance
                step_y_in_agent_view = step[0] - agent_y + self.__view_distance
                
                if step_x_in_agent_view < 0 or step_x_in_agent_view >= observation_size\
                or step_y_in_agent_view < 0 or step_y_in_agent_view >= observation_size:
                    #oob step -> agent disapeared
                    continue
                                
                astar_map[i, step_x_in_agent_view, step_y_in_agent_view] = 1
            
        #pathlength map
        path = agent.path_to_goal_on_self_knowledge(self.__finder,start_position=agent.get_current_position(), end_position=agent.get_goal_position())
        pathlength = len(path)
        
        for step in path:
            #! step coordinates are not in the right order
            step_x_in_agent_view = step[1] - agent_x + self.__view_distance
            step_y_in_agent_view = step[0] - agent_y + self.__view_distance
            if step_x_in_agent_view < 0 or step_x_in_agent_view >= observation_size\
                or step_y_in_agent_view < 0 or step_y_in_agent_view >= observation_size:
                    #out of observation step -> stop
                    break
            
            pathlength_map[step_x_in_agent_view, step_y_in_agent_view] = pathlength
            pathlength -= 1
            pathlength_map = preprocessing.normalize(pathlength_map)
        
        #compute the state of the agent
        state = np.array([wall_map, other_agent_map, pathlength_map], dtype=np.float32)
        state = np.concatenate((state, astar_map), axis=0, dtype=np.float32)
        return state