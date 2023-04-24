from World import *
from MapfObserver import *
import gym
from gym import spaces
from enum import Enum
from matplotlib import pyplot as plt
import imageio
import pygame
import seaborn as sns
from pathfinding.finder.a_star import AStarFinder
import random

class MapfENV(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps":4}
    
    
    def __init__(self, map_generator, observer, num_agents, render_mode=None, render_fps=None):
        
        self.windows_x_size = 1600 # pygame window size
        self.windows_y_size = self.windows_x_size / 2
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.window = None
        self.clock = None
        self.__map_generator:MapGenerator = map_generator
        self.__observer:MapfObserver = observer
        self.__num_agents = num_agents
        self.__world:World = None
        self.__world = None
        self.__finder = AStarFinder()
    
    def _get_info(self):
        return "no info implemented"
    
    def reset(self):        
        # Generate the world
        self.__world = World(self.__map_generator, self.__num_agents, communication_distance=self.__observer.get_communication_distance(), view_distance=self.__observer.get_view_distance())
        self.__observer.set_world(self.__world)
        self.__observer.set_finder(self.__finder)
        
        observations = self.__observer.observe()
        
        if self.render_mode == 'human':
            self._render_frame(observations)
            
        return observations
        
        
    def step(self, joint_action):
        """
        take the joint action and send it to the world to execute it

        Args:
            - joint_action : Dict[agent_id:{'movement': movement_value, 'ask_communication': ask_communication_value}]

        returns
            - observations : the dictionnary of agents observation after the execution of the joint_action
            - rewards : the dictionnary of agents rewards after the execution of the joint_action
            - dones : the dictionnary of agents objective status (has it reach it's goal)
            - information about the world
        """
        movement_actions = {}
        agents_asking_for_communication = []

        for agent_id, agent_action in joint_action.items():
            
            movement_actions.update({agent_id: MovementAction(agent_action['movement'])})
            
            if agent_action['ask_communication'] == 1:
                agents_asking_for_communication.append(agent_id)

        #updates the agent knowledge if they are asking communication
        self.__world.compute_communication(agents_asking_for_communication)
        
        #simulate the movement_actions on the world
        status_update, valid_positions = self.__world.simulate_joint_action(movement_actions)
    
        self.__world.clear_maps()
        rewards = {agent_id: None for agent_id in movement_actions.keys()}
        
        #update agents based on status_update and valid_positions & compute agents reward
        for agent_id in movement_actions.keys():
            agent = self.__world.get_agents()[agent_id]
            new_position = valid_positions[agent_id]
            new_status = status_update[agent_id]
            agent.move(new_position, new_status)
            agent.update_knowledge(self.__world.agent_world_view(agent))

            reward = self.__compute_reward(agent, agent_id in agents_asking_for_communication)
            rewards.update({agent_id: reward})
        
        #update the agent positions on the world
        self.__world.update_maps()        
        
        observations = self.__observer.observe()
        
        dones = {agent_id: self.__world.get_agent(agent_id).has_reached_goal() for agent_id in movement_actions.keys()}
        
        if self.render_mode == "human" and False in dones.values():
            self._render_frame(observations)
        
        return observations, rewards, dones, "info"
    
    def __compute_reward(self, agent:Agent, asking_for_knowledge_communication:bool):
        
        """compute the reward to associate to the agent depending on it's status
        
        Arguments:
            Agent: the agent
        
        Returns:
            int : the reward of the agent
        """
        agent_status:Status = agent.get_status()
        
        if agent_status is Status.REACHING_GOAL:
            return 5 + -0.3 #goal reward + movement cost
        
        if agent_status is Status.STANDED_STILL and agent.on_goal():
            return 0 #no movement and on goal
        
        if agent_status is Status.COLLIDED:
            return -2 + -0.3 #movement and collision
        
        if agent_status is Status.MOVED:
            return -0.3
        
        else:
            return ValueError
    
    
    ############################################################################ WORLD RENDER ########################################################################     
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self, observations):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.windows_x_size, self.windows_y_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        def general_map():
            canvas = pygame.Surface((self.windows_x_size/2, self.windows_y_size))
        # the origo point (0, 0) is in the top left corner of the window. The x coordinates increase to the right, and the y coordinates increase downwards
        
            canvas.fill((255,255,255))
            pix_square_size = (
                (self.windows_x_size/2) / self.__world.get_size()[0]
            ) # the size of a single grid square in pixels
            
            for wall in self.__world.get_walls_positions():
                pygame.draw.rect(
                    canvas,
                    (0,0,0),
                    pygame.Rect(
                        (pix_square_size * wall[1], pix_square_size * wall[0]),
                        (pix_square_size, pix_square_size),
                    ),
                )
                
            for agent in self.__world.get_agents().values():
                if not agent.on_goal():
                    #draw objectif
                    pygame.draw.rect(
                        canvas,
                        tuple(a*255 for a in sns.color_palette("Spectral", as_cmap=True)(agent.get_id() * (1 / self.__num_agents))),
                        pygame.Rect(
                            (pix_square_size * agent.get_goal_position()[1], pix_square_size * agent.get_goal_position()[0]),
                            (pix_square_size, pix_square_size),
                        ),
                    )
                else:
                    #draw wall
                    pygame.draw.rect(
                        canvas,
                        (0,0,0),
                        pygame.Rect(
                            (pix_square_size * agent.get_goal_position()[1], pix_square_size * agent.get_goal_position()[0]),
                            (pix_square_size, pix_square_size),
                        ),
                    )
                    
            for agent in self.__world.get_agents().values():
                if agent.get_status() is Status.COLLIDED:
                    pygame.draw.circle(
                        canvas,
                        (255,0,0),
                        ((agent.get_current_position()[1] + 0.5) * pix_square_size, (agent.get_current_position()[0] + 0.5) * pix_square_size),
                        pix_square_size / 2.5,
                    )
                
                pygame.draw.circle(
                    canvas,
                    tuple(a*255 for a in sns.color_palette("Spectral", as_cmap=True)(agent.get_id() * (1 / self.__num_agents))),
                    ((agent.get_current_position()[1] + 0.5) * pix_square_size, (agent.get_current_position()[0] + 0.5) * pix_square_size),
                    pix_square_size / 3,
                )
                    
            for x in range(self.__world.get_size()[0] + 1):
                pygame.draw.line(
                    canvas,
                    0,
                    (0, pix_square_size * x),
                    (self.windows_x_size/2, pix_square_size * x),
                    width=3,
                )
                pygame.draw.line(
                    canvas,
                    0,
                    (pix_square_size * x, 0),
                    (pix_square_size * x, self.windows_y_size),
                    width=3,
                )
            return canvas 
       
        
        #agents knowledge canvas
        def agent_knowledge(agent, agent_observation):

            pix_square_size = ((self.windows_x_size/4) / self.__world.get_size()[0]) # the size of a single grid square in pixels
            
            agent_knowledge = pygame.Surface((self.windows_x_size/4, self.windows_y_size/2))
            agent_knowledge.fill((255,255,255))

            def get_walls_positions(agent:Agent):
                return np.argwhere(agent.get_world_knowledge() == 1)

            for wall in get_walls_positions(agent):
                pygame.draw.rect(
                agent_knowledge,
                (0,0,0),
                pygame.Rect(
                    (pix_square_size * wall[1], pix_square_size * wall[0]),
                    (pix_square_size, pix_square_size),
                ),
                )

            
            if not agent.has_reached_goal():
                    path = agent.path_to_goal_on_self_knowledge(self.__finder)

                    pygame.draw.rect(
                            agent_knowledge,
                            tuple(a*255 for a in sns.color_palette("Spectral", as_cmap=True)(agent.get_id() * (1 / self.__num_agents))),
                            pygame.Rect(
                                (pix_square_size * agent.get_goal_position()[1], pix_square_size * agent.get_goal_position()[0]),
                                (pix_square_size, pix_square_size),
                            ),
                        )

                    for step_x, step_y in path[1:-1]:
                        pygame.draw.rect(
                            agent_knowledge,
                            tuple(a*255 for a in sns.color_palette("Spectral", as_cmap=True)(agent.get_id() * (1 / self.__num_agents))),
                            pygame.Rect(
                                (pix_square_size * step_x, pix_square_size * step_y),
                                (pix_square_size, pix_square_size),
                            ),
                        )


            pygame.draw.circle(
                    agent_knowledge,
                    tuple(a*255 for a in sns.color_palette("Spectral", as_cmap=True)(agent.get_id() * (1 / self.__num_agents))),
                    ((agent.get_current_position()[1] + 0.5) * pix_square_size, (agent.get_current_position()[0] + 0.5) * pix_square_size),
                    pix_square_size / 3,
                )

            for x in range(self.__world.get_size()[0] + 1):
                pygame.draw.line(
                agent_knowledge,
                0,
                (0, pix_square_size * x),
                (self.windows_x_size/2, pix_square_size * x),
                width=3,
                )
                pygame.draw.line(
                agent_knowledge,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.windows_y_size),
                width=3,
                )

            
            #lines around the knowledge map
            pygame.draw.line(
                agent_knowledge,
                (0,0,255),
                (0, 0),
                (self.windows_x_size/2, 0),
                width=3,
            )
            pygame.draw.line(
                agent_knowledge,
                (0,0,255),
                (0, pix_square_size * self.__world.get_size()[0]),
                (self.windows_x_size/2, pix_square_size * self.__world.get_size()[0]),
                width=3,
            )
            pygame.draw.line(
                agent_knowledge,
                (0,0,255),
                (0, 0),
                (0, self.windows_y_size),
                width=3,
                )
            pygame.draw.line(
                agent_knowledge,
                (0,0,255),
                (pix_square_size * self.__world.get_size()[0], 0),
                (pix_square_size * self.__world.get_size()[0], self.windows_y_size),
                width=3,
                )
            return agent_knowledge
        
        def agent_walls(agent:Agent, agent_observation):
            agent_x, agent_y = agent.get_current_position()
            walls_observed = agent_observation[0]
            agents_observed = agent_observation[1]
            pathlength_map = agent_observation[2]
            astar_maps = agent_observation[3:]
            agent_world_knowledge = agent.get_world_knowledge()

            world_shape = agent_world_knowledge.shape[0]
            observations_shape = agent_observation[0].shape[0]

            pix_square_size = ((self.windows_x_size/4) / observations_shape) # the size of a single grid square in pixels
            
            agent_obs = pygame.Surface((self.windows_x_size/4, self.windows_y_size/2))
            agent_obs.fill((255,255,255))

            for i in range(observations_shape):
                for j in range(observations_shape):
                    knowledge_i, knwoledge_j = i+agent_x-observations_shape//2, j+agent_y-observations_shape//2
                    if not (knowledge_i < 0 or knowledge_i >= world_shape or \
                        knwoledge_j < 0 or knwoledge_j >= world_shape): #not oob observation

                        if agent.get_world_knowledge()[i+agent_x-observations_shape//2][j+agent_y-observations_shape//2]:
                            pygame.draw.rect(
                                agent_obs,
                                (125,125,125),
                                pygame.Rect(
                                    (pix_square_size * j, pix_square_size * i),
                                    (pix_square_size, pix_square_size),
                                ),
                            )

                    if walls_observed[i][j]:
                        pygame.draw.rect(
                            agent_obs,
                            (0,0,0),
                            pygame.Rect(
                                (pix_square_size * j, pix_square_size * i),
                                (pix_square_size, pix_square_size),
                            ),
                        )

                    if pathlength_map[i][j] > 0:
                       pygame.draw.rect(
                            agent_obs,
                            (0,255,0),
                            pygame.Rect(
                                (pix_square_size * j, pix_square_size * i),
                                (pix_square_size, pix_square_size),
                            ),
                        )
                    
                    for astar_map in astar_maps:
                        #print(astar_map)
                        if astar_map[i][j]:
                            pygame.draw.rect(
                                agent_obs,
                                (125,0,0),
                                pygame.Rect(
                                    (pix_square_size * j, pix_square_size * i),
                                    (pix_square_size, pix_square_size),
                                ),
                            )
                       
                    if agents_observed[i][j]:
                        pygame.draw.circle(
                            agent_obs,
                            (255,0,0),
                            ((j + 0.5) * pix_square_size, (i + 0.5) * pix_square_size),
                            pix_square_size / 3,
                        )


            return agent_obs
            
        active_agents = self.__world.get_active_agents()
        if active_agents != []:
            agent = active_agents[0]
            agent1_knowledge_map = agent_knowledge(agent, observations[agent.get_id()])
            agent1_walls_obs_map = agent_walls(agent, observations[agent.get_id()])
                
            if len(active_agents) > 1:
                agent = active_agents[1]
                agent2_knowledge_map = agent_knowledge(agent, observations[agent.get_id()])
                agent2_walls_obs_map = agent_walls(agent, observations[agent.get_id()])
            else:
                agent2_knowledge_map = agent1_knowledge_map
                agent2_walls_obs_map = agent1_walls_obs_map


        canvas = general_map()
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            self.window.blit(agent1_knowledge_map, (800,0))
            self.window.blit(agent1_walls_obs_map, (1200,0))
            self.window.blit(agent2_knowledge_map, (800,400))
            self.window.blit(agent2_walls_obs_map, (1200,400))
            pygame.event.pump()
            pygame.display.update()
            if self.render_fps is None:
                self.clock.tick(self.metadata["render_fps"])
            else:
                self.clock.tick(self.render_fps)
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1,0,2)
            )
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
    
    def get_AStar_next_action(self,agent_id):
        if not self.__world.get_agent(agent_id).has_reached_goal():
            path = self.__world.get_agent(agent_id).path_to_goal_on_self_knowledge(finder=self.__finder)
            if path != []:
                init_pos = path[0]
                next_pos = path[1]
                dir = (next_pos[1]-init_pos[1],next_pos[0]-init_pos[0])
                return direction_to_movement_action(dir).value
        
        return random.choice(list(MovementAction)).value

