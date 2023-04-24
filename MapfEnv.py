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

class Rewards(Enum):
    COLLISION = -2
    STANDING_STILL = -0.3
    ACTION = -0.3
    GOAl = 5
    BROADCAST = - 2

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

        self.movement_action_space = spaces.Box(low = 0, high = len(MovementAction) -1, shape=(self.__num_agents,1), dtype=np.uint8)       
        self.broadcast_action_space = spaces.Box(low = 0, high = len(BroadcastAction) -1, shape=(self.__num_agents,1), dtype=np.uint8)
    
    def _get_info(self):
        return {agent_id: np.linalg.norm(tuple_minus(self.__world.get_agents()[agent_id].get_current_position(),self.__world.get_agents()[agent_id].get_goal_position()), ord=1) for agent_id, agent in self.__world.get_agents().items()}
        
    def reset(self):        
        # Generate the world
        self.__world = World(self.__map_generator, self.__num_agents, broadcast_distance=self.__observer.get_broadcast_distance(), view_distance=self.__observer.get_view_distance())
        self.__observer.set_world(self.__world)
        
        observation = self.__observer.observe()
        
        if self.render_mode == 'human':
            self._render_frame()
            
        return observation
        
        
    def step(self, movement_action, broadcast_action):
        """
        take the joint action and send it to the world to execute it

        Args:
            - joint_action : array[(movement_action, broadcast_action)]

        returns
            - observations : the dictionnary of agents observation after the execution of the joint_action
            - rewards : the dictionnary of agents rewards after the execution of the joint_action
            - dones : the dictionnary of agents objective status (has it reach it's goal)
            - information about the world
        """
        movement_actions = {}
        broadcast_actions = {}

        for id, move_action in enumerate(movement_action):
            movement_actions.update({id+1: MovementAction(move_action)})
        
        for id, broadcast_action in enumerate(broadcast_action):
            broadcast_actions.update({id+1: BroadcastAction(broadcast_action)}) 

        #broadcast the agent_knowledge if agent decided to broadcast
        self.__world.compute_broadcast(broadcast_actions)
        
        #simulate the movement_actions on the world
        status_update, valid_positions = self.__world.simulate_joint_action(movement_actions)
    
        self.__world.clear_maps()
        rewards = {agent_id: None for agent_id in movement_actions.keys()}
        dones   = {agent_id: False for agent_id in movement_actions.keys()}
        
        #update agents based on status_update and valid_positions & compute agents reward
        for agent_id in movement_actions.keys():
            agent = self.__world.get_agents()[agent_id]
            
            new_position = valid_positions[agent_id]
            new_status = status_update[agent_id]
            agent.move(new_position, new_status)
            if agent.has_reached_goal():
                dones.update({agent_id: True})
            agent.update_knowledge(self.__world.agent_world_observation(agent))

            reward = self.__compute_reward(new_status,broadcast_actions[agent_id])
            rewards.update({agent_id: reward})
        
        #update the agent positions on the world
        self.__world.update_maps()        
        
        observations = {agent_id: self.__observer.observe_agent(agent_id) for agent_id in movement_actions.keys()}
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observations, rewards, dones, "info"
    
    def __compute_reward(self, status:Status, broadcast_action:BroadcastAction):
                   
        if status is Status.STANDED_STILL:
            reward = Rewards.STANDING_STILL.value
            
        if status is Status.MOVED:
            reward = Rewards.ACTION.value
        
        if status is Status.COLLIDED:
            reward = Rewards.ACTION.value + Rewards.COLLISION.value
            
        if status is Status.MOVED_TO_GOAL:
            reward = Rewards.ACTION.value + Rewards.GOAl.value
        
        if broadcast_action is BroadcastAction.ON:
            return reward + Rewards.BROADCAST.value
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.windows_x_size, self.windows_y_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
            
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
            if not agent.has_reached_goal():
                pygame.draw.rect(
                    canvas,
                    tuple(a*255 for a in sns.color_palette("Spectral", as_cmap=True)(agent.get_id() * (1 / self.__num_agents))),
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

        #agents knowledge canvas
        pix_square_size = ((self.windows_x_size/4) / self.__world.get_size()[0]) # the size of a single grid square in pixels
        
        agent_1 = self.__world.get_agent(1)
        agent_1_knowledge = pygame.Surface((self.windows_x_size/4, self.windows_y_size/2))
        agent_1_knowledge.fill((255,255,255))

        def get_walls_positions(agent:Agent):
            return np.argwhere(agent.get_world_knowledge() == 1)

        for wall in get_walls_positions(agent_1):
            pygame.draw.rect(
            agent_1_knowledge,
            (0,0,0),
            pygame.Rect(
                (pix_square_size * wall[1], pix_square_size * wall[0]),
                (pix_square_size, pix_square_size),
            ),
            )

        
        if not agent_1.has_reached_goal():
                path = agent_1.path_to_goal(self.__finder)

                pygame.draw.rect(
                        agent_1_knowledge,
                        tuple(a*255 for a in sns.color_palette("Spectral", as_cmap=True)(agent_1.get_id() * (1 / self.__num_agents))),
                        pygame.Rect(
                            (pix_square_size * agent_1.get_goal_position()[1], pix_square_size * agent_1.get_goal_position()[0]),
                            (pix_square_size, pix_square_size),
                        ),
                    )

                for step_x, step_y in path[1:-1]:
                    pygame.draw.rect(
                        agent_1_knowledge,
                        tuple(a*255 for a in sns.color_palette("Spectral", as_cmap=True)(agent_1.get_id() * (1 / self.__num_agents))),
                        pygame.Rect(
                            (pix_square_size * step_x, pix_square_size * step_y),
                            (pix_square_size, pix_square_size),
                        ),
                    )


        
        pygame.draw.circle(
                agent_1_knowledge,
                tuple(a*255 for a in sns.color_palette("Spectral", as_cmap=True)(1 * (1 / self.__num_agents))),
                ((agent_1.get_current_position()[1] + 0.5) * pix_square_size, (agent_1.get_current_position()[0] + 0.5) * pix_square_size),
                pix_square_size / 3,
            )

        for x in range(self.__world.get_size()[0] + 1):
            pygame.draw.line(
            agent_1_knowledge,
            0,
            (0, pix_square_size * x),
            (self.windows_x_size/2, pix_square_size * x),
            width=3,
            )
            pygame.draw.line(
            agent_1_knowledge,
            0,
            (pix_square_size * x, 0),
            (pix_square_size * x, self.windows_y_size),
            width=3,
            )

        
        #lines around the knowledge map
        pygame.draw.line(
            agent_1_knowledge,
            (0,0,255),
            (0, 0),
            (self.windows_x_size/2, 0),
            width=3,
        )
        pygame.draw.line(
            agent_1_knowledge,
            (0,0,255),
            (0, pix_square_size * self.__world.get_size()[0]),
            (self.windows_x_size/2, pix_square_size * self.__world.get_size()[0]),
            width=3,
        )
        pygame.draw.line(
            agent_1_knowledge,
            (0,0,255),
            (0, 0),
            (0, self.windows_y_size),
            width=3,
            )
        pygame.draw.line(
            agent_1_knowledge,
            (0,0,255),
            (pix_square_size * self.__world.get_size()[0], 0),
            (pix_square_size * self.__world.get_size()[0], self.windows_y_size),
            width=3,
            )
        
        agent_2 = self.__world.get_agent(2)
        agent_2_knowledge = pygame.Surface((self.windows_x_size/4, self.windows_y_size/2))
        agent_2_knowledge.fill((255,255,255))

        for wall in get_walls_positions(agent_2):
            pygame.draw.rect(
            agent_2_knowledge,
            (0,0,0),
            pygame.Rect(
                (pix_square_size * wall[1], pix_square_size * wall[0]),
                (pix_square_size, pix_square_size),
            ),
            )

        
        if not agent_2.has_reached_goal():
                path = agent_2.path_to_goal(self.__finder)

                pygame.draw.rect(
                        agent_2_knowledge,
                        tuple(a*255 for a in sns.color_palette("Spectral", as_cmap=True)(2 * (1 / self.__num_agents))),
                        pygame.Rect(
                            (pix_square_size * agent_2.get_goal_position()[1], pix_square_size * agent_2.get_goal_position()[0]),
                            (pix_square_size, pix_square_size),
                        ),
                    )

                for step_x, step_y in path[1:-1]:
                    pygame.draw.rect(
                        agent_2_knowledge,
                        tuple(a*255 for a in sns.color_palette("Spectral", as_cmap=True)(2 * (1 / self.__num_agents))),
                        pygame.Rect(
                            (pix_square_size * step_x, pix_square_size * step_y),
                            (pix_square_size, pix_square_size),
                        ),
                    )


        for x in range(self.__world.get_size()[0] + 1):
            pygame.draw.line(
                agent_2_knowledge,
                0,
                (0, pix_square_size * x),
                (self.windows_x_size/2, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                agent_2_knowledge,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.windows_y_size),
                width=3,
            )

        pygame.draw.circle(
                agent_2_knowledge,
                tuple(a*255 for a in sns.color_palette("Spectral", as_cmap=True)(2 * (1 / self.__num_agents))),
                ((agent_2.get_current_position()[1] + 0.5) * pix_square_size, (agent_2.get_current_position()[0] + 0.5) * pix_square_size),
                pix_square_size / 3,
            )
        #lines around the knowledge map
        pygame.draw.line(
            agent_2_knowledge,
            (0,0,255),
            (0, 0),
            (self.windows_x_size/2, 0),
            width=3,
        )
        pygame.draw.line(
            agent_2_knowledge,
            (0,0,255),
            (0, pix_square_size * self.__world.get_size()[0]),
            (self.windows_x_size/2, pix_square_size * self.__world.get_size()[0]),
            width=3,
        )
        pygame.draw.line(
            agent_2_knowledge,
            (0,0,255),
            (0, 0),
            (0, self.windows_y_size),
            width=3,
            )
        pygame.draw.line(
            agent_2_knowledge,
            (0,0,255),
            (pix_square_size * self.__world.get_size()[0], 0),
            (pix_square_size * self.__world.get_size()[0], self.windows_y_size),
            width=3,
            )

        agent_3 = self.__world.get_agent(3)
        agent_3_knowledge = pygame.Surface((self.windows_x_size/4, self.windows_y_size/2))
        agent_3_knowledge.fill((255,255,255))

        for wall in get_walls_positions(agent_3):
            pygame.draw.rect(
            agent_3_knowledge,
            (0,0,0),
            pygame.Rect(
                (pix_square_size * wall[1], pix_square_size * wall[0]),
                (pix_square_size, pix_square_size),
            ),
            )

        
        if not agent_3.has_reached_goal():
                path = agent_3.path_to_goal(self.__finder)

                pygame.draw.rect(
                        agent_3_knowledge,
                        tuple(a*255 for a in sns.color_palette("Spectral", as_cmap=True)(3 * (1 / self.__num_agents))),
                        pygame.Rect(
                            (pix_square_size * agent_3.get_goal_position()[1], pix_square_size * agent_3.get_goal_position()[0]),
                            (pix_square_size, pix_square_size),
                        ),
                    )

                for step_x, step_y in path[1:-1]:
                    pygame.draw.rect(
                        agent_3_knowledge,
                        tuple(a*255 for a in sns.color_palette("Spectral", as_cmap=True)(3 * (1 / self.__num_agents))),
                        pygame.Rect(
                            (pix_square_size * step_x, pix_square_size * step_y),
                            (pix_square_size, pix_square_size),
                        ),
                    )
        
        for x in range(self.__world.get_size()[0] + 1):
            pygame.draw.line(
                agent_3_knowledge,
                0,
                (0, pix_square_size * x),
                (self.windows_x_size/2, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                agent_3_knowledge,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.windows_y_size),
                width=3,
            )
        pygame.draw.circle(
                agent_3_knowledge,
                tuple(a*255 for a in sns.color_palette("Spectral", as_cmap=True)(3 * (1 / self.__num_agents))),
                ((agent_3.get_current_position()[1] + 0.5) * pix_square_size, (agent_3.get_current_position()[0] + 0.5) * pix_square_size),
                pix_square_size / 3,
            )
        
        #lines around the knowledge map
        pygame.draw.line(
            agent_3_knowledge,
            (0,0,255),
            (0, 0),
            (self.windows_x_size/2, 0),
            width=3,
        )
        pygame.draw.line(
            agent_3_knowledge,
            (0,0,255),
            (0, pix_square_size * self.__world.get_size()[0]),
            (self.windows_x_size/2, pix_square_size * self.__world.get_size()[0]),
            width=3,
        )
        pygame.draw.line(
            agent_3_knowledge,
            (0,0,255),
            (0, 0),
            (0, self.windows_y_size),
            width=3,
            )
        pygame.draw.line(
            agent_3_knowledge,
            (0,0,255),
            (pix_square_size * self.__world.get_size()[0], 0),
            (pix_square_size * self.__world.get_size()[0], self.windows_y_size),
            width=3,
            )


        agent_4 = self.__world.get_agent(4)
        agent_4_knowledge = pygame.Surface((self.windows_x_size/4, self.windows_y_size/2))
        agent_4_knowledge.fill((255,255,255))

        for wall in get_walls_positions(agent_4):
            pygame.draw.rect(
            agent_4_knowledge,
            (0,0,0),
            pygame.Rect(
                (pix_square_size * wall[1], pix_square_size * wall[0]),
                (pix_square_size, pix_square_size),
            ),
            )

        if not agent_4.has_reached_goal():
                path = agent_4.path_to_goal(self.__finder)

                pygame.draw.rect(
                        agent_4_knowledge,
                        tuple(a*255 for a in sns.color_palette("Spectral", as_cmap=True)(4 * (1 / self.__num_agents))),
                        pygame.Rect(
                            (pix_square_size * agent_4.get_goal_position()[1], pix_square_size * agent_4.get_goal_position()[0]),
                            (pix_square_size, pix_square_size),
                        ),
                    )

                for step_x, step_y in path[1:-1]:
                    pygame.draw.rect(
                        agent_4_knowledge,
                        tuple(a*255 for a in sns.color_palette("Spectral", as_cmap=True)(4 * (1 / self.__num_agents))),
                        pygame.Rect(
                            (pix_square_size * step_x, pix_square_size * step_y),
                            (pix_square_size, pix_square_size),
                        ),
                    )
        
        for x in range(self.__world.get_size()[0] + 1):
            pygame.draw.line(
                agent_4_knowledge,
                0,
                (0, pix_square_size * x),
                (self.windows_x_size/2, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                agent_4_knowledge,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.windows_y_size),
                width=3,
            )

        pygame.draw.circle(
                agent_4_knowledge,
                tuple(a*255 for a in sns.color_palette("Spectral", as_cmap=True)(4 * (1 / self.__num_agents))),
                ((agent_4.get_current_position()[1] + 0.5) * pix_square_size, (agent_4.get_current_position()[0] + 0.5) * pix_square_size),
                pix_square_size / 3,
            )
        
         #lines around the knowledge map
        pygame.draw.line(
            agent_4_knowledge,
            (0,0,255),
            (0, 0),
            (self.windows_x_size/2, 0),
            width=3,
        )
        pygame.draw.line(
            agent_4_knowledge,
            (0,0,255),
            (0, pix_square_size * self.__world.get_size()[0]),
            (self.windows_x_size/2, pix_square_size * self.__world.get_size()[0]),
            width=3,
        )
        pygame.draw.line(
            agent_4_knowledge,
            (0,0,255),
            (0, 0),
            (0, self.windows_y_size),
            width=3,
            )
        pygame.draw.line(
            agent_4_knowledge,
            (0,0,255),
            (pix_square_size * self.__world.get_size()[0], 0),
            (pix_square_size * self.__world.get_size()[0], self.windows_y_size),
            width=3,
            )
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())

            self.window.blit(agent_1_knowledge, (800,0))
            self.window.blit(agent_2_knowledge, (1200,0))
            self.window.blit(agent_3_knowledge, (800,400))
            self.window.blit(agent_4_knowledge, (1200,400))
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
            path = self.__world.get_agent(agent_id).path_to_goal(finder=self.__finder)
            if path != []:
                init_pos = path[0]
                next_pos = path[1]
                dir = (next_pos[1]-init_pos[1],next_pos[0]-init_pos[0])
                return direction_to_movement_action(dir).value
        
        return random.choice(list(MovementAction)).value

