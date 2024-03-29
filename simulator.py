"""
This is the Simulator module. All sensors variables and physics handling are implemented here. You don't need to read this
code to understand this assignment. If you came here because of a bug, please let us know about it.

Attributes:
    int width,height: size of window and simulation space. Changing these variable may cause unexpected behaviour!
    bool evaluate: If GUI will be rendered or not.
"""
import math
import random
import sys
import csv

import numpy as np
import pygame
from PIL import Image
from pygame.color import THECOLORS

import pymunk as pymunk
from pymunk import Vec2d
from pymunk.pygame_util import draw
from trigonometry import *
from controller1.controller import Controller, State
from controller2.controller import Controller as Controller2
from controller2.controller import State as State2
import datetime, time

# PyGame screen dimensions
width = 1000
height = 700

# Gui Flag
show_simulation = True

# Velocity constants shared among classes in this module
VEL_MIN = 10
VEL_MAX = 200
INIT_VELOCITY = 10

# Colision types required by pumunk
CAR_COLLISION_TYPE = 500
CHECKPOINT_COLLISION_TYPE = 501
RADAR_COLLISION_TYPE = 502
BOMB_COLLISION_TYPE = 503

#images to be used
car_image = "assets/car.png"
bot_image = "assets/bot.png"
bomb_image = "assets/bomb.png"

explosion_shape = pygame.image.load('assets/explosion.png')


def block_print():
    """
    Disables print because I know you didn't compile chipmunk to disable debugs warnings
    """
    sys.stdout = None


def enable_print():
    sys.stdout = sys.__stdout__


def get_point_from_rgb_list(x: int, y: int, image_vector: 'list of rgb elements') -> list:
    """
    Get point from pygame rgb vector using pymunk coordinates
    :param x: pymunk x coordinate
    :param y: pymunk y coordinate
    :param image_vector: pygame Image
    :return: an list that contain a rgb vector
    """
    pos = (height - y - 1) * width + x
    try:
        return image_vector[pos]
    except IndexError:
        return image_vector[0]


# The following five functions are collision functions used by pymunk collision handler

def mark_checkpoint(game, attribute):
    """
    Function to handle pymunk collision calls for checkpoints
    """
    for shape in attribute.shapes:
        if shape.collision_type != CHECKPOINT_COLLISION_TYPE:
            shape.class_bound.mark_checkpoint(game, attribute)

    return True


def crash_penalty(game, attribute):
    """
    Activate crash penalty for cars which have collided
    """
    for shape in attribute.shapes:
        shape.class_bound.crash_penalty(game, attribute)
    return True


def disable_carsh_penality(game, attribute):
    """
    Remove penalties once separated
    """
    for shape in attribute.shapes:
        shape.class_bound.disable_crash_penalty(game, attribute)
    return True


def detect_obstacle(game, attribute):
    """
    Add obstacle to processing list
    """
    for shape in attribute.shapes:
        if shape.collision_type == RADAR_COLLISION_TYPE:
            shape.class_bound.detect_obstacle(game, attribute)
    return True


def remove_obstacle_from_list(game, attribute):
    """
    Remove obstacles from processing list
    """
    for shape in attribute.shapes:
        if shape.collision_type == RADAR_COLLISION_TYPE:
            shape.class_bound.remove_obstacle_from_list(game, attribute)
    return True


class Background(pygame.sprite.Sprite):
    def __init__(self, image_path: str, location: (int, int)):
        """
        # Simple class extension to make easier to draw background

        :param image_path:
        :param location:
        """
        pygame.sprite.Sprite.__init__(self)  # call Sprite initializer
        self.image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(self.image, (width, height))
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = location


class CarShape(pymunk.Poly):
    def __init__(self, body, rectangle, car_bound):
        """
        Simple extension of Pymunk's Poly class

        :param body: Pymunk body object
        :param rectangle: Polygon coordinates forming and rectangle
        :param car_bound: Car object which shape is beeing linked to.
        """
        self.class_bound = car_bound
        super().__init__(body, rectangle)
        pass


class BombShape(pymunk.Circle):
    def __init__(self, body, radius, car_bound):
        """
        Simple extension of Pymunk's Poly class

        :param body: Pymunk body object
        :param radius: Radius of circle
        :param car_bound: Car object which shape is beeing linked to.
        """
        self.class_bound = car_bound
        super().__init__(body, radius)
        pass


class _Car:

    ACTION_LIST = (1, 2, 3, 4, 5)

    def __init__(self, space, track, position, track_rgb, off_track_color, checkpoints, radar_collision_type,
                 img_path,screen=None):
        """
        This class is used to represent a Car in the Simulation, it handles movement and sensors.

        :param space: pymunk space
        :param track: Track object which contains
        :param position: tuple containing x,y coordinates
        :param track_rgb: rgb vector containing track information
        :param off_track_color: collor that represent off_track in track_rgb
        :param checkpoints: list of checkpoints the game is using
        :param radar_collision_type: integer representing
        :param screen: Pymunk screen
        """

        # Initializing class variables
        self.enemy_detected = False
        self.car_img_shape = pygame.image.load(img_path)
        self.radar_collision_type = radar_collision_type
        self.position = position
        self.punctuation = 0
        self.grass_penalty = 0
        self.max_checkpoints = len(track.checkpoints)
        self.space = space
        self.track = track
        self.off_track_color = off_track_color
        self.track_rgb = track_rgb
        self._create_new_car_body()
        self.current_checkpoint_distance = 0
        self.last_checkpoint_distance = 0
        self.checkpoints = checkpoints
        self.current_checkpoint = 0
        self.obstacle_body_position_angle = 0
        self.point_in_front = 0
        self.car_direction = 0
        self.crashed = False
        self.on_track = True
        self.obstacle_distance = 0
        self.bodies_around = []
        self.frame_count = 0
        self.first = True
        self.checkpoint_sensor = 0
        self.off_track_color = (85, 186, 54, 255)
        self.on_track_color = (129, 126, 37, 255)
        self.ice_track_color = (104, 165, 243, 255)
        self.obs = 0
        self.crash_timer = 0

        # Add Pymunk's collision handling if that wasn't done before
        self.space.add_collision_handler(CAR_COLLISION_TYPE, CHECKPOINT_COLLISION_TYPE, begin=mark_checkpoint)
        self.space.add_collision_handler(CAR_COLLISION_TYPE, CAR_COLLISION_TYPE, begin=crash_penalty,
                                         separate=disable_carsh_penality)
        self.space.add_collision_handler(RADAR_COLLISION_TYPE, CAR_COLLISION_TYPE, begin=detect_obstacle,
                                         separate=remove_obstacle_from_list)
        self.space.add_collision_handler(CAR_COLLISION_TYPE, BOMB_COLLISION_TYPE, begin=crash_penalty,
                                         separate=disable_carsh_penality)
        self.space.add_collision_handler(RADAR_COLLISION_TYPE, BOMB_COLLISION_TYPE, begin=detect_obstacle,
                                         separate=remove_obstacle_from_list)

        # Creates screen instance if in evaluate mode
        if show_simulation:
            self.screen = screen

    def car_step(self, action: int):

        """
        Prepare car for next frame

        :param action: Integer indicating car's action
        """

        self.frame_count += 1

        vel = self.car_body.vel

        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        t_x, t_y = self.car_body.position
        current_color = get_point_from_rgb_list(int(t_x), int(t_y), self.track_rgb)

        if current_color == self.ice_track_color:
            self.on_track = 2
            # Turning actions
            if action == 1:  # Turn right.
                self.car_body.angle -= .1
            elif action == 2:  # Turn left.
                self.car_body.angle += .1
        else:
            # Turning actions
            if action == 1:  # Turn right.
                self.car_body.angle -= .2
            elif action == 2:  # Turn left.
                self.car_body.angle += .2



        # Check if car is not out of bounds
        if t_x >= width or t_y >= height or t_x <= 0 or t_y <= 0:
            self.on_track = 0
            vel = VEL_MIN
        if self.crashed:
            self.crash_timer = 5
            # self.crashed is updated by self.disable_crash_penalty method
        elif (current_color == self.on_track_color) or (current_color == (127, 125, 29, 255)):
            # Updates velocity normally
            self.on_track = 1
            if action == 3:
                if (vel + 20) <= VEL_MAX:  # accelerate
                    vel += 20
                else:
                    vel = VEL_MAX
            elif action == 4:
                if (vel - 50) >= VEL_MIN:  # break
                    vel -= 50
                else:
                    vel = VEL_MIN
        elif current_color == self.off_track_color or (current_color == (106, 188, 44, 255)):
            # Check if car is off track and if so gradually slows its current velocity
            self.on_track = 0
            self.grass_penalty += 1
            if (vel - 30) >= VEL_MIN:
                vel -= 30
            else:
                vel = VEL_MIN

        if self.crash_timer > 0:
            self.crash_timer -= 1
            vel = VEL_MIN

        # Updates car velocity vector
        self.car_body.velocity = vel * driving_direction
        self.car_body.vel = vel

        # Updates car angle
        self.car_direction = driving_direction

        pass

    def draw(self):
        """
        Draws car on screen according to angle and velocity
        """

        #Poll events
        event = pygame.event.poll()
        
        if show_simulation:
            p = self.car_body.position
            # Correct p because pygame crazy coordinate system
            p = Vec2d(p.x, height - p.y)

            # Transform image to right size and flips it
            new_img = pygame.transform.scale(self.car_img_shape, (40, 20))
            new_img = pygame.transform.flip(new_img, True, False)

            # Rotates image and place it at cars position
            angle_degrees = math.degrees(self.car_body.angle) + 180
            new_img = pygame.transform.rotate(new_img, angle_degrees)
            offset = Vec2d(new_img.get_size()) / 2.
            p = p - offset

            # Renders Image
            self.screen.blit(new_img, p)

    def _create_new_car_body(self):
        """
        Setups a bunch of pymunk's configurations
        """
        rectangle = [(-20, -10), (-20, 10), (20, 10), (20, -10)]
        self.car_body = pymunk.Body(100, pymunk.inf)
        self.car_body.position = self.position[0], self.position[1]
        self.car_shape = CarShape(self.car_body, rectangle, self)
        self.car_shape.color = THECOLORS["green"]
        self.car_shape.elasticity = 0
        self.car_body.angle = self.track.angle_of_cars
        self.car_shape.collision_type = CAR_COLLISION_TYPE
        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.apply_impulse(driving_direction)
        self.space.add(self.car_body, self.car_shape)
        self.car_body.vel = VEL_MIN
        self.car_radar = self._create_radar_sensor(self.car_body)

    def _create_radar_sensor(self, car_body: pymunk.Body) -> pymunk.Circle:
        """
        Creates invisible radar attached to car
        """
        c_shape = pymunk.Circle(car_body, 100)
        setattr(c_shape, 'class_bound', self)
        c_shape.sensor = True
        c_shape.collision_type = RADAR_COLLISION_TYPE
        c_shape.ignore_draw = True
        self.space.add(c_shape)
        return c_shape

    # noinspection PyUnusedLocal
    def detect_obstacle(self, game, attribute):
        """
        Function called by pymunk's collision handling function witch detects nearby objects. This is called when the
        car's sensor is overlapping another object.
        """
        for shape in attribute.shapes:
            if shape is self.car_body:
                return True

        for shape in attribute.shapes:
            if shape is not self.car_radar:
                self.bodies_around.append(shape)
                # print(self.bodies_around)
        return True

    # noinspection PyUnusedLocal
    def remove_obstacle_from_list(self, game, attribute):
        """
        Function called by pymunk's collision handling function witch detects nearby objects. This is called when the
        car'sm sensor is no more overlapping an object.
        """
        for shape in attribute.shapes:
            if shape.collision_type == CAR_COLLISION_TYPE or shape.collision_type == BOMB_COLLISION_TYPE:
                if shape in self.bodies_around:
                    self.bodies_around.remove(shape)
        return True

    def compute_nearest_body(self) -> (float, float):
        """
        Search witch object within car's radar is the nearest
        :return: Obstacle distance and Obstacle angle from the nearest obstacle
        """

        if not self.bodies_around:
            return -1, 0, 0

        if not any(bodies.collision_type == CAR_COLLISION_TYPE for bodies in self.bodies_around):
            return -1, 0, 1

        vec = Vec2d(1, 0).rotated(self.car_body.angle)

        # Prepares vectors
        displacement = np.array([self.car_body.position[0], self.car_body.position[1]])
        robot_center = np.array([0.0, 0.0])
        robot_antenna = np.array([vec[0], vec[1]])
        robot_center = np.add(robot_center, displacement)
        robot_antenna = np.add(robot_antenna, displacement)

        # Initialize variable with infinite
        nearest_distance = float('inf')
        for shape in self.bodies_around:
            if shape.collision_type == CAR_COLLISION_TYPE:
                new_distance = math.hypot(shape.body.position[0] - self.car_body.position[0],
                                          shape.body.position[1] - self.car_body.position[1])
                # Checks if it is the nearest body found yet
                if new_distance <= nearest_distance:
                    # Updates distance
                    self.obstacle_distance = new_distance

                    # Updates Angle
                    enemy_pos = np.array([shape.body.position[0], shape.body.position[1]])
                    enemy_pos_origin = np.subtract(enemy_pos, robot_center)
                    antena_pos_origin = np.subtract(robot_antenna, robot_center)
                    self.obstacle_body_position_angle = angle_between_with_quadrant(enemy_pos_origin, antena_pos_origin)
                    self.obstacle_body_position_angle = rad2deg(self.obstacle_body_position_angle)

        return self.obstacle_distance, self.obstacle_body_position_angle, 1

    def compute_nearest_bomb(self) -> tuple:
        """
        Search witch bombs within car's radar is the nearest
        :return: Obstacle distance and Obstacle angle from the nearest obstacle
        """

        if not self.bodies_around:
            return -1, 0, 0

        if not any(bodies.collision_type == BOMB_COLLISION_TYPE for bodies in self.bodies_around):
            return -1, 0, 1

        vec = Vec2d(1, 0).rotated(self.car_body.angle)

        # Prepares vectors
        displacement = np.array([self.car_body.position[0], self.car_body.position[1]])
        robot_center = np.array([0.0, 0.0])
        robot_antenna = np.array([vec[0], vec[1]])
        robot_center = np.add(robot_center, displacement)
        robot_antenna = np.add(robot_antenna, displacement)

        # Initialize variable with infinite
        nearest_distance = float('inf')
        for shape in self.bodies_around:
            if shape.collision_type == BOMB_COLLISION_TYPE:
                new_distance = math.hypot(shape.body.position[0] - self.car_body.position[0],
                                          shape.body.position[1] - self.car_body.position[1])
                # Checks if it is the nearest body found yet
                if new_distance <= nearest_distance:
                    # Updates distance
                    obstacle_distance = new_distance

                    # Updates Angle
                    enemy_pos = np.array([shape.body.position[0], shape.body.position[1]])
                    enemy_pos_origin = np.subtract(enemy_pos, robot_center)
                    antena_pos_origin = np.subtract(robot_antenna, robot_center)
                    obstacle_body_position_angle = angle_between_with_quadrant(enemy_pos_origin, antena_pos_origin)
                    obstacle_body_position_angle = rad2deg(obstacle_body_position_angle)

        return obstacle_distance, obstacle_body_position_angle, 1

    # noinspection PyUnusedLocal
    def crash_penalty(self, game, attribute):
        """
        Function called by pymunk's collision handling function witch detects if two cars have collided. This is
        called when the cars just started touching
        """
        self.crashed = True
        return True

    # noinspection PyUnusedLocal
    def disable_crash_penalty(self, game, attribute):
        """
        Function called by pymunk's collision handling function witch detects if two cars have collided. This is
        called when the cars just started touching
        """
        self.crashed = False
        return True

    def _draw_track_sensor(self, rotated_p: (float, float)):
        """
        Draws a track sensor
        :param rotated_p: The point where the sensor has ended
        """
        if show_simulation:
            pygame.draw.line(self.screen, (255, 255, 255), rotated_p,
                             (self.car_body.position[0], height - self.car_body.position[1]))

    def reset(self):
        """
        Reset car variables to prepare for another simulation
        """
        self.first = True
        self.frame_count = 0
        self.grass_penalty = 0
        self.punctuation = 0
        self.current_checkpoint = 0
        self.space.remove(self.car_body, self.car_shape, self.car_radar)
        self._create_new_car_body()
        self.bodies_around = []
        self.crashed = False

    @property
    def sensors(self) -> list:

        """
        :return: List which contains (in order):
        track_distance_left: 1-100
        track_distance_center: 1-100
        track_distance_right: 1-100
        on_track: 0 or 1
        checkpoint_distance: 0-???
        car_velocity: 10-200
        enemy_distance: -1 or 0-???
        position_angle: -180 to 180
        enemy_detected: 0 or 1
        checkpoint: 1 or 0
        """

        # Default values
        self.obstacle_distance = -1
        self.obstacle_body_position_angle = 0

        # Gets track readings
        x, y = self.car_body.position
        readings = self._get_sonar_readings(x, y, self.car_body.angle)

        # Gets checkpoint distances
        checkpoint_distance = distance(self.checkpoints[self.current_checkpoint], self.car_body.position)
        self.current_checkpoint_distance = checkpoint_distance

        # Gets Enemy detection sensors
        self.obstacle_distance, self.obstacle_body_position_angle, enemy_detected = self.compute_nearest_body()

        bomb_distance, bomb_position_angle, bomb_detected = self.compute_nearest_bomb()

        # Checks checkpoint sensor:
        if self.checkpoint_sensor == 1:
            checkpoint = 1
            self.checkpoint_sensor = 0
        else:
            checkpoint = 0

        if self.obs == self.on_track_color:
            incoming_track = 1
        elif self.obs == self.ice_track_color:
            incoming_track = 2
        else:
            incoming_track = 0

        if self.on_track > 0 and incoming_track == 0:
            incoming_track = self.on_track

        sensors = [readings[0], readings[1], readings[2], int(self.on_track), checkpoint_distance,
                   self.car_body.vel, self.obstacle_distance, self.obstacle_body_position_angle, enemy_detected,
                   checkpoint, incoming_track, bomb_detected, bomb_distance, bomb_position_angle]

        return sensors

    def _get_sonar_readings(self, x: float, y: float, angle):
        """
        Get track readings given car position and angle
        :param x: car x
        :param y: car y
        :param angle: car angle
        :return:
        """

        readings = []
        # Make our arms.
        arm_left = self.make_sonar_arm(x, y)
        arm_middle = arm_left
        arm_right = arm_left

        # Rotate them and get readings.
        readings.append(self._get_arm_distance(arm_left, x, y, angle, 0.75))
        readings.append(self._get_arm_distance(arm_middle, x, y, angle, 0, center=True))
        readings.append(self._get_arm_distance(arm_right, x, y, angle, -0.75))

        if show_simulation:
            pygame.display.update()

        return readings

    def _get_arm_distance(self, arm: list, x: float, y: float, angle: float, offset: float, center=False) -> int:
        # Used to count the distance.
        """
        Calculates track sensor values
        :param arm: sonar arm
        :param x: car x
        :param y: car y
        :param angle: car angle
        :param offset: used for correction
        :param center: True if this is the central track sensor, false otherwise
        :return: Distance ranging from 1-100
        """
        i = 0
        rotated_p = self.get_rotated_point(
            x, y, arm[0][0], arm[0][1], angle + offset
        )

        if center:
            self.obs = get_point_from_rgb_list(rotated_p[0], height - rotated_p[1], self.track_rgb)

        # Look at each point and see if we've hit something.

        for point in arm:
            i += 1

            # Move the point to the right spot.
            rotated_p = self.get_rotated_point(
                x, y, point[0], point[1], angle + offset
            )
            if i == 1 and center:
                self.point_in_front = rotated_p

            # Check if we've hit something. Return the current i (distance)
            # if we did.
            if rotated_p[0] <= 0 or rotated_p[1] <= 0 \
                    or rotated_p[0] >= width or rotated_p[1] >= height:
                self._draw_track_sensor(rotated_p)
                return i  # Sensor is off the screen.
            else:
                obs = get_point_from_rgb_list(rotated_p[0], height - rotated_p[1], self.track_rgb)

                if self.get_track_or_not(obs) != 0:
                    self._draw_track_sensor(rotated_p)
                    return i
                elif center:
                    self.obs = obs

        self._draw_track_sensor(rotated_p)
        return i

    @staticmethod
    def make_sonar_arm(x: float, y: float) -> list:
        """
        :return: list of points
        """
        spread = 1  # Default spread.
        arm_distance = 5  # Gap before first sensor.
        arm_points = []
        # Make an arm. We build it flat because we'll rotate it about the
        # center later.
        for i in range(1, 101):
            arm_points.append((arm_distance + x + (spread * i), y))

        return arm_points

    @staticmethod
    def get_rotated_point(x_1: float, y_1: float, x_2: float, y_2: float, radians: float) -> (float, float):
        """
        Computes rotated points for sonar arms

        :param x_1: car x
        :param y_1: car y
        :param x_2: sonar arm x
        :param y_2: sonar arm y
        :param radians: sonar arm angle
        :return: Point for sonar arm
        """
        # Rotate x_2, y_2 around x_1, y_1 by angle.
        x_change = (x_2 - x_1) * math.cos(radians) + \
                   (y_2 - y_1) * math.sin(radians)
        y_change = (y_1 - y_2) * math.cos(radians) - \
                   (x_1 - x_2) * math.sin(radians)
        new_x = x_change + x_1
        new_y = height - (y_change + y_1)
        return int(new_x), int(new_y)

    def get_track_or_not(self, reading: list) -> int:
        """
        Checks if car is in track

        :param reading: A list representing a color
        :return: 0 or 1 indicating true or false
        """
        #if reading != self.on_track_color and reading != self.ice_track_color:
        if reading != self.ice_track_color and reading != self.on_track_color and reading != (127, 125, 29, 255):
            return 1
        else:
            return 0

    @property
    def score(self) -> int:
        """
        :return: Car's score
        """
        return self.punctuation - (10*self.grass_penalty) - self.current_checkpoint_distance

    # noinspection PyUnusedLocal
    def mark_checkpoint(self, game, attribute):
        """
        Function called by pymunk's collision handling function witch detects if a car has crossed a checkpoint. This is
        called when a car starts touching a checkpoint.
        """
        for shape in attribute.shapes:
            if self.checkpoints[self.current_checkpoint] is shape:
                self.current_checkpoint += 1
                self.punctuation += 500
                self.frame_count = 0
                self.checkpoint_sensor = 1
        if self.current_checkpoint >= self.max_checkpoints:
            self.current_checkpoint = 0
            for checkpoint in self.checkpoints:
                checkpoint.color = THECOLORS['yellow']
        return True


class _Bot(_Car):
    """
    Class to control bot behavior.
    """
    def __init__(self, space, track, position, track_rgb, off_track_color, checkpoints, car_collision_type, bot_type,
                 img_path, screen=None):
        super().__init__(space, track, position, track_rgb, off_track_color, checkpoints, car_collision_type, img_path,
                         screen)
        self.bot_type = bot_type
        self.curr = 0
        self.steps = 0
        self.can_break = True
        self.break_count = 0
        pass

    def ninja_bot(self,sensors):
        dis_left = sensors[0]
        dis_front = sensors[1]
        dis_right = sensors[2]
        velocity = sensors[5]

        epsilon = 30
        vel_max = 120

        if math.fabs(dis_left - dis_right) < epsilon:
            self.can_break = True
            self.break_count = 0

        if dis_front > (velocity/2) and (velocity+20) < vel_max: #can accel
            action = 3 #accel
        elif dis_front < (velocity/2) and self.can_break: #need to brake
            action = 4 #break
            if self.break_count < 2:
                self.break_count += 1
            else:
                self.can_break = False
        elif (dis_right - dis_left) > epsilon:
            action = 1 #right
        elif (dis_left - dis_right) > epsilon:
            action = 2 #left
        else:
            action = 5 #no change

        return action

    # noinspection PyUnusedLocal
    def dumb_bot(self, sensors):
        if self.steps == 5:
            self.curr = random.randint(1, 5)
            self.steps = 0
        else:
            self.steps += 1

        return self.curr

    def safe_bot(self, sensors):

        dis_left = sensors[0]
        dis_front = sensors[1]
        dis_right = sensors[2]
        epsilon = 10

        if (dis_right - dis_left) > epsilon:
            action = 1  # right
        elif (dis_left - dis_right) > epsilon:
            action = 2  # left
        elif dis_front < 10:
            action = 4  # break
        else:
            action = 3  # accel

        return action

    def custom_bot(self, sensors):
        raise NotImplementedError("You must implement custom_bot")

    def choose_action(self):
        sensors = self.sensors
        if self.bot_type == 'dumb_bot':
            return self.dumb_bot(sensors)
        elif self.bot_type == 'safe_bot':
            return self.safe_bot(sensors)
        elif self.bot_type == 'ninja_bot':
            return self.ninja_bot(sensors)
        elif self.bot_type == 'custom_bot':
            return self.custom_bot(sensors)
        else:
            raise ValueError("This bot type is not recognized")

    # noinspection PyUnusedLocal
    def car_step(self, action):
        action = self.choose_action()
        super().car_step(action)


class _ParkedBot(_Car):
    def __init__(self, space, track, position, track_rgb, off_track_color, checkpoints, car_collision_type,
                 img_path, screen=None):
        """
        A car that does nothing but possibly collide with other cars
        """
        super().__init__(space, track, position, track_rgb, off_track_color, checkpoints, car_collision_type,
                         img_path,screen)
        pass

    def _create_new_car_body(self):
        rectangle = [(-20, -10), (-20, 10), (20, 10), (20, -10)]
        self.car_body = pymunk.Body(100, pymunk.inf)
        self.car_body.position = self.position[0][0], self.position[0][1]
        self.car_shape = CarShape(self.car_body, rectangle, self)
        self.car_shape.color = THECOLORS["green"]
        self.car_shape.elasticity = 1.0
        self.car_body.angle = self.position[1]
        self.car_shape.collision_type = CAR_COLLISION_TYPE
        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.apply_impulse(driving_direction)
        self.space.add(self.car_body, self.car_shape)
        self.car_body.vel = INIT_VELOCITY

    def car_step(self, action):
        self.car_body.velocity = (0, 0)

    def reset(self):
        self.punctuation = 0
        self.current_checkpoint = 0
        self.space.remove(self.car_body, self.car_shape)
        self._create_new_car_body()
        self.bodies_around = []


class _Bomb(_Car):
    def __init__(self, space, track, position, track_rgb, off_track_color, checkpoints, car_collision_type,
                 img_path, screen=None):
        """
        A bomb, I'm reusing the car code because I'm lazy
        """

        self.exploded = False
        self.explosion_effect_counter = 10
        self.explosion_effect = True

        super().__init__(space, track, position, track_rgb, off_track_color, checkpoints, car_collision_type,
                         img_path, screen)

    def _create_new_car_body(self):
        """
         
        """
        self.car_body = pymunk.Body(100, pymunk.inf)
        self.car_body.position = self.position[0][0], self.position[0][1]
        self.car_shape = BombShape(self.car_body, 20, self)
        self.car_shape.color = THECOLORS["black"]
        self.car_shape.ignore_draw = True
        self.car_body.angle = self.position[1]
        self.car_shape.collision_type = BOMB_COLLISION_TYPE
        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.apply_impulse(driving_direction)
        self.space.add(self.car_body, self.car_shape)
        self.car_body.vel = INIT_VELOCITY

    def car_step(self, action: int):
        if self.crashed:
            self.car_img_shape = explosion_shape
            try:
                self.space.remove(self.car_body)
                self.space.remove(self.car_shape)
                self.exploded = True
            except KeyError:
                pass

    def reset(self):
        try:
            self.space.remove(self.car_body)
            self.space.remove(self.car_shape)
        except KeyError:
            pass

    def draw(self):
        """
        Draws bomb on screen according to angle and velocity
        """
        if show_simulation:
            p = self.car_body.position
            # Correct p because pygame crazy coordinate system
            p = Vec2d(p.x, height - p.y)

            # Transform image to right size and flips it
            new_img = pygame.transform.scale(self.car_img_shape, (38, 38))
            new_img = pygame.transform.flip(new_img, True, False)

            # Rotates image and place it at cars position
            angle_degrees = math.degrees(self.car_body.angle) + 180
            new_img = pygame.transform.rotate(new_img, angle_degrees)
            offset = Vec2d(new_img.get_size()) / 2.
            p = p - offset

            print(self.explosion_effect_counter)
            if self.exploded:
                self.explosion_effect_counter -= 1
                if self.explosion_effect_counter <= 0:
                    self.explosion_effect = False

            # Renders Image
            self.screen.blit(new_img, p)


class Simulation:
    def __init__(self, track, bot_type):
        """
        Handles simulation and GUI
        :param track: Track object witch configures the scenario
        :param bot_type: Type of bot to be alongside user, can be set to None for no bot
        """

        # Initialize GUI if requested
        if show_simulation:
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            self.clock = pygame.time.Clock()
            self.screen.set_alpha(None)

        # Initialize class variables
        self.last_checkpoint_distance = 0
        self.current_checkpoint_distance = 0
        self.current_checkpoint = 0
        self.track = track
        self.frame_count = 0
        self.on_track = True
        self.bot_type = bot_type

        self.global_track = Background(self.track.display_img_path, [0, 0])
        self.crashed_single_time = False
        self.max_steps = 3000
        self.crashed = False
        self.punctuation = 0
        self.environment = []
        self.force_switch = True

        # Physics stuff.
        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0., 0.)

        self.checkpoints = []

        for pair_of_points in track.checkpoints:
            self.checkpoints.append(self._create_checkpoint(pair_of_points))
        # Record steps.
        self.num_steps = 0

        # More GUI stuff
        if show_simulation:
            self.screen.fill(THECOLORS["black"])
            self.screen.blit(self.global_track.image, self.global_track.rect)
            draw(self.screen)
            pygame.display.flip()

        # Track variables
        self.image = Image.open(self.track.mask_img_path)
        self.image = self.image.resize((width, height))
        self.track_rgb = list(self.image.getdata())
        self.off_track_color = (84, 174, 50, 255)
        self.on_track_color = (163, 123, 75, 255)
        self.ice_track_color = (13, 193, 217, 255)

        # GUI stuff
        if show_simulation:
            game_screen = self.screen
        else:
            game_screen = None

        self.bombs = []

        # Test bomb
        # self.bomb1 = _Bomb(self.space, self.track, (220,210), self.track_rgb,
        #                    self.off_track_color,
        #                    self.checkpoints, 1001, bot_image, screen=game_screen)

        # Creates player car
        self.car1 = _Car(self.space, self.track, self.track.car1_position, self.track_rgb, self.off_track_color,
                         self.checkpoints, 1000, car_image, screen=game_screen)

        # Get sample of 4 random bombs positions
        random_bombs_position = random.sample(self.track.bombs, 4)

        for i in range(0, len(random_bombs_position)):
            self.bombs.append(
                _Bomb(self.space, self.track, random_bombs_position[i], self.track_rgb,
                      self.off_track_color,
                      self.checkpoints, 1001, bomb_image, screen=game_screen))

        # Initialize bots
        if bot_type is not None:
            if bot_type == 'player2':
                self.car_bot = _Car(self.space, self.track, self.track.car2_position, self.track_rgb, self.off_track_color,
                         self.checkpoints, 1000, bot_image, screen=game_screen)
            else:
                self.car_bot = _Bot(self.space, self.track, self.track.car2_position,
                                    self.track_rgb, self.off_track_color,
                                    self.checkpoints, 1000, bot_type, bot_image, screen=game_screen)

        self.game_objects = [i for i in self.bombs]
        self.game_objects.append(self.car1)

        # Add bots to
        if bot_type is not None and bot_type != 'parked_bots':
            self.game_objects.append(self.car_bot)

    def learn(self, controller: Controller, number_of_episodes: int) -> None:

        episode_count = 0
        best_score = float('-inf')
        while episode_count < number_of_episodes:
            self.reset()

            # Initial step
            sensors = self.frame_step(5)

            new_state = State(sensors)
            old_state = State(sensors)

            frame_number = 0
            start = True
            while frame_number <= self.track.episode_length:
               
                action = controller.take_action(new_state, episode_count)
                sensors = self.frame_step(action)
                old_state = new_state
                new_state = State(sensors)
                #print(new_state.get_current_state())
                #print(old_state.get_current_state())
                reward = controller.compute_reward(new_state, old_state, action, frame_number, False)
                
                controller.update_q(new_state, old_state, action, reward, False)

                frame_number += 1

            print("episode",episode_count,"score",self.car1.score)
            

            if self.car1.score >= best_score:
                best_score = self.car1.score
                output = "./params/%s_%d.txt" % (datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S'),
                                                 episode_count)
                controller.q_table.save(output)

            
            file_name = "./dados/Episodio Pontuacao.csv"
            file = open(file_name, 'a', newline='')
            writer = csv.writer(file, delimiter=';')
            row = []
            #row.append(episode_count)
            row.append(round(self.car1.score))
            writer.writerow(row)
            

            episode_count += 1

        #output = "./params/%s_final.txt" % datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
        output = "./params/final.txt"
        controller.q_table.save(output)

        pass

    def evaluate(self, controller: Controller) -> None:
        frame_number = 0
        sensors = self.frame_step(5)
        while frame_number <= self.track.episode_length:
            state = State(sensors)
            q_values = [controller.q_table.get_q_value(state, i) for i in (1, 2, 3, 4, 5)]
            action = q_values.index(max(q_values)) + 1
            sensors = self.frame_step(action)
            frame_number += 1
        print("score",self.car1.score)
        pass

    def evaluate_comp(self, player_1: Controller, player_2: Controller) -> None:
        frame_number = 0
        self.car1.car_step(5)
        self.car_bot.car_step(5)

        sensors_player_1 = self.car1.sensors
        sensors_player_2 = self.car_bot.sensors

        self.comp_frame_step()

        while frame_number <= self.track.episode_length:
            state_1 = State(sensors_player_1)
            state_2 = State2(sensors_player_2)

            q_values_1 = [player_1.q_table.get_q_value(state_1, i) for i in _Car.ACTION_LIST]
            q_values_2 = [player_2.q_table.get_q_value(state_2, i) for i in _Car.ACTION_LIST]

            action_1 = q_values_1.index(max(q_values_1)) + 1
            action_2 = q_values_2.index(max(q_values_2)) + 1

            self.car1.car_step(action_1)
            self.car_bot.car_step(action_2)

            self.comp_frame_step()

            sensors_player_1 = self.car1.sensors
            sensors_player_2 = self.car_bot.sensors

            frame_number = frame_number + 1
        pass

    def _create_checkpoint(self, pair_of_points: ((float, float), (float, float)), color='yellow')-> pymunk.Poly:
        """
        Create checkpoint sensor
        :param pair_of_points: Coordinates of checkpoint segment
        :param color: Color of checkpoint
        :return: checkpoint shape
        """

        c_body = pymunk.Body(1, 1)
        c_shape = pymunk.Poly(c_body, pair_of_points)
        c_shape.sensor = True
        c_shape.elasticity = 100
        c_shape.color = THECOLORS[color]
        c_shape.collision_type = CHECKPOINT_COLLISION_TYPE
        self.space.add(c_body, c_shape)
        return c_shape

    def reset(self):
        """
        Resets simulation
        """
        self.car1.reset()
        if self.bot_type is not None and self.bot_type != 'parked_bots':
            self.car_bot.reset()
        for bomb in self.bombs:
            bomb.reset()

        # GUI stuff
        if show_simulation:
            game_screen = self.screen
        else:
            game_screen = None

        random_bombs_position = random.sample(self.track.bombs, 4)
        self.bombs = []
        for i in range(0, len(random_bombs_position)):
            self.bombs.append(
                _Bomb(self.space, self.track, random_bombs_position[i], self.track_rgb,
                      self.off_track_color,
                      self.checkpoints, 1001, bot_image, screen=game_screen))

            # self.car_bot.reset()

    def frame_step(self, action: int) -> list:
        """
        Advances simulation by one frame.
        :param action: Action to be given to player's car
        :return: sensors player's car acquired by advancing frame.
        """
        self.frame_count += 1

        self.car1.car_step(action)
        if self.bot_type is not None and self.bot_type != 'parked_bots':
            self.car_bot.car_step(0)

        for parked_car in self.bombs:
            parked_car.car_step(0)

        # self.bomb1.car_step(0)
        self.space.step(1. / 10)

        if show_simulation:
            block_print()
            self._draw_screen()
            enable_print()
        sensors = self.car1.sensors
        if self.bot_type is not None and self.bot_type != 'parked_bots':
            # noinspection PyStatementEffect
            self.car_bot.sensors

        return sensors

    def comp_frame_step(self):

        for parked_car in self.bombs:
            parked_car.car_step(0)

        self.space.step(1. / 10)
        if show_simulation:
            block_print()
            self._draw_screen()
            enable_print()
        pass

    def _draw_screen(self):
        self.screen.fill(THECOLORS["black"])
        self.screen.blit(self.global_track.image, self.global_track.rect)
        self.car1.draw()
        if self.bot_type is not None and self.bot_type != 'parked_bots':
            self.car_bot.draw()
        for bomb in self.bombs:
            print(bomb.explosion_effect)
            if not bomb.exploded:
                bomb.draw()
            elif bomb.explosion_effect:
                bomb.draw()

        draw(self.screen, self.space)
        # pygame.display.flip()
        self.clock.tick()

        pass
