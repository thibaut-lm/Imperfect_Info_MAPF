from Agent import *
from MapGenerator import *
import copy
import numpy as np
from skimage.draw import line

from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

class World:
    
    def __init__(self, map_generator:MapGenerator, num_agents:int, communication_distance, view_distance):
        self.__map_generator:MapGenerator = map_generator
        
        self.__agents = {}
        self.__num_agents = num_agents
        
        self.__state_map = None
        self.__goal_map = None

        self.__view_distance = view_distance
        self.__communication_distance = communication_distance
        
        self.reset()
        
    def reset(self):
        """
        Reset the world:
         - generate a new map
         - initiate agents on the new map
        """
        
        # Generate the map
        self.__state_map = self.__map_generator.generate()
        self.__goal_map = np.zeros((self.__state_map.shape[0], self.__state_map.shape[1])).astype(int)
        
        # Generate the agents
        self.__agents = {id: copy.deepcopy(Agent(id)) for id in range(1, self.__num_agents + 1)}
        
        # Initiate agents & map
        self.__init_agent_and_goals()
        
        # Update agent's communication neighboors
        self.update_communication_neighboors()
    
    def agent_world_view(self, agent:Agent):
        """Get the np.array of shape (world_shape) describing the world walls view aournd a specific agent
            - if view[i][j] == -1 : the agent detects a wall at position [i][j]

        Args:
            agent (Agent): the agent we want to know it's current view of the world walls

        Returns:
            np.array : the current view of the world walls of the agent
        """
        
        # Get the agent positions, world walls & init view
        agent_x, agent_y = agent.get_current_position()
        walls_map = self.get_walls_map()
        agent_view = np.zeros(walls_map.shape).astype(int)
        
        # Get the valid ranges where the agent can see (i.e ranges without out of band positions)
        agent_x_view_range, agent_y_view_range = self.distance_ranges(agent_x, agent_y, self.__view_distance)
    
        for x in agent_x_view_range:
            for y in agent_y_view_range:
                
                # Check if valid position[x][y] is in view of the agent and if it's a wall update agent_view
                if self.in_view(agent_x, agent_y, x, y):
                    agent_view[x,y] = walls_map[x,y]
        
        return agent_view
    
    def get_size(self):
        return self.__state_map.shape[0], self.__state_map.shape[1]
    
    def get_state_map(self):
        return self.__state_map

    def get_walls_positions(self):
        return np.argwhere(self.__state_map == -1)
    
    def get_walls_map(self):
        return (self.__state_map == -1).astype(int) 
    
    def get_goal_map(self):
        return self.__goal_map
    
    def get_agents(self):
        return self.__agents
    
    def get_agent(self, id):
        return self.__agents[id]
    
    def __free_positions(self):
        return np.argwhere(self.__state_map == 0)
    
    def clear_maps(self):
        self.__state_map[self.__state_map > 0] = 0 # Remove agent from them state_map
        self.__goal_map[self.__goal_map != 0] = 0 # Remove agent goals from the goal_map
        
    def update_maps(self):
        """
        Update the world maps based on it's agent positions.
        """
        for agent in self.get_agents().values():             
            self.__state_map[agent.get_current_position()] = agent.get_id()
            self.__goal_map[agent.get_goal_position()] = agent.get_id()
        
        self.update_communication_neighboors()

    def distance_ranges(self, x, y, distance):
        """Return valid x and y ranges of size (2* distance + 1) around the given [x][y] position
            - if the position is out of the world the it'll not be in the range.

        Args:
            x (int): x coordinate
            y (int): y coordinate
            distance (int): distance around the given [x][y] coordinate we want to compute the ranges from

        Returns:
            (x_range, y_range): the x and y ranges computed
        """
        x_range = (max(0, x - distance), min(self.__state_map.shape[0], x + distance+1))
        y_range = (max(0, y - distance), min(self.__state_map.shape[1], y + distance+1))
        return range(x_range[0], x_range[1]), range(y_range[0], y_range[1])
    
    def in_communication_range(self, a_x, a_y, b_x, b_y):
        """Check if the two positions given are in communication range in the current world

        Args:
            a_x (int): first agent x coordinate
            a_y (int): first agent y coordinate
            b_x (int): second agent x coordinate
            b_y (int): second agent y coordinate

        Returns:
            bool: True if the agent are in communication range
        """
        
        # Get the x and y valid ranges of agent a with a distance of communication 
        a_x_comm_range , a_y_comm_range = self.distance_ranges(a_x, a_y, distance=self.__communication_distance)
        
        return b_x in a_x_comm_range and b_y in a_y_comm_range # Check if agent b coordinates are in those ranges

    def in_view(self, a_x, a_y, b_x, b_y):
        """Check if two positions can see each other
            - (i.e) no obstacle between them

        Args:
            a_x (int): first agent x coordinate
            a_y (int): first agent y coordinate
            b_x (int): second agent x coordinate
            b_y (int): second agent y coordinate

        Returns:
            bool: True if there is no obstacle between the two given positions
        """
        return not self.obstacle_between(a_x, a_y, b_x, b_y)
        
    def obstacle_between(self,a_x, a_y, b_x, b_y):
        """Check is there is an obstacle between two given positions

        Args:
            a_x (int): first agent x coordinate
            a_y (int): first agent y coordinate
            b_x (int): second agent x coordinate
            b_y (int): second agent y coordinate

        Returns:
            bool: True if there is an obstacle detected between the two given positions
        """
        # Trace a line from [a_x][a_y] to [b_x][b_y]
        rr,cc = line(a_x, a_y, b_x, b_y)
        
        # Check if cell [a_x][a_y] is cell [b_x][b_y] or the cell [b_x][b_y] is next cell [a_x][a_y]
        if len(rr) <= 2 :
            return False
        
        # Check if there is any obstacle in the line between a and b
        return True in(self.__state_map[rr[1:-1], cc[1:-1]] != 0)

    def update_communication_neighboors(self):
        """ 
        Updates the communication neighboors of the agents based on their current positions.
        """
        
        # Update information for each agent
        agents = self.__agents.values()
        for agent in agents:
            
            # Reset communication neighboors
            agent.reset_communication_neighboors()
            
            for neighbor in agents:
                
                # Agent do not communicate with himself
                if agent.get_id() == neighbor.get_id(): continue            
                
                # Get the agent and neighbor positions
                agent_x, agent_y = agent.get_current_position()
                neighbor_x, neighbor_y = neighbor.get_current_position()
                
                # Update the communication neighboors of agent is they are in communication_range & can see each other
                if self.in_communication_range(agent_x, agent_y, neighbor_x, neighbor_y):
                    if self.in_view(agent_x, agent_y, neighbor_x, neighbor_y):
                        agent.add_communication_neighboors(neighbor)

    def get_active_agents(self):
        return [agent for agent_id, agent in self.__agents.items() if not agent.has_reached_goal()]
    
    def __init_agent_and_goals(self):
        # Init grid and finder
        grid = Grid(matrix=1-self.__state_map)
        finder = AStarFinder()
        
        for agent_id, agent in self.__agents.items():
            
            # Select a valid start and goal position
            valid_positions = False
            while not valid_positions:
                
                #clear the grid
                grid.cleanup()
                
                # Select a start and a goal position
                free_positions = self.__free_positions()
                rand_indexes = np.random.choice(len(free_positions), size = 2, replace = False)
                start_position, end_position = [tuple(free_positions[index]) for index in rand_indexes]
                
                # Check if there is a path
                start = grid.node(start_position[1], start_position[0])
                end =  grid.node(end_position[1], end_position[0])
                path, runs = finder.find_path(start, end, grid)
                
                if len(path) != 0:
                    valid_positions = True
                    
                    # Update the agent
                    agent.move(start_position)
                    agent.set_goal(end_position)
                    
                    # Initiate the agents knowledge
                    agent.init_knowledge(self.agent_world_view(agent))
                    
                    # Update the maps
                    self.__state_map[start_position] = agent_id
                    self.__goal_map[start_position] = agent_id
    
    def simulate_joint_action(self, joint_action: list[(MovementAction, AskingCommunicationAction)]):
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
                        status_update.update({current_agent_id: Status.REACHING_GOAL})
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
            if not self.__agents[current_agent_id].has_reached_goal() and wanted_positions[current_agent_id] == self.__agents[current_agent_id].get_goal_position():
                status_update.update({current_agent_id: Status.REACHING_GOAL})
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
        if self.oob(wanted_position) :
            return True
        # no wall neither oob
        return False

    def oob(self, position):
        return position[0] < 0 or position[0] >= self.__state_map.shape[0] \
            or position[1] < 0 or position[1] >= self.__state_map.shape[1]
    
    def compute_communication(self, agents_asking_for_communication:list[int]):
        """Compute the communication to the agent asking for communication from it's communication neighbors

        Args:
            agents_asking_for_communication (list[int]): list of agents ids asking for communication
        """
        for agent_id in agents_asking_for_communication:
            agent = self.__agents[agent_id]
            
            # Get it's communication neighboors
            communication_neighboors = agent.get_communication_neighboors()
            for communication_neighboor in communication_neighboors:
                    agent.update_knowledge(communication_neighboor.get_world_knowledge())
