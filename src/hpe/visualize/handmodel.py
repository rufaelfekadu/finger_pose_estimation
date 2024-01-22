import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import math


class HandModel3D:
    def __init__(self, angles_file):
        self.angles_file = angles_file
        self.angle_frames = self.read_angle_frames()
        self.origin = (0, -1, 0)

    def read_angle_frames(self):
        with open(self.angles_file, 'r') as file:
            return [[float(angle) for angle in line.split()] for line in file.readlines()]

    def calculate_point(self, base_point, angles, distances):
        x = base_point[0]
        y = base_point[1]
        z = base_point[2]

        for angle, distance in zip(angles, distances):
            x += distance * math.cos(math.radians(angle))
            y += distance * math.sin(math.radians(angle))

        return x, y, z
    
    def draw_palm(self):
        hand_color = (0.6, 0.8, 1.0)  # Adjust color if needed
        glColor3fv(hand_color)
        
        palm_points = [
            (-0.4, 0.3, 0.0),  # pinky finger base 
            (-0.15, 0.4, 0.0),  # ring finger base
            (0.4, 0.4, 0.0),  # index finger base
            (0.35, -0.4, 0.0),  #  thumb base
            (-0.35, -0.45, 0.0), #  
            (-0.4, 0.3, 0.0)  # Example: Bottom of the palm
            # Add more points as needed to draw the palm shape
        ]
        
        # Draw lines to connect the palm points
        glBegin(GL_LINES)
        for i in range(len(palm_points)-1):
            glVertex3fv(palm_points[i])  # Start from the wrist point (base)
            glVertex3fv(palm_points[i+1])  # Draw lines to each static point
        glEnd()

        # Draw vertices (points) at the palm points
        glPointSize(5.0)  # Set the size of the points
        glColor3fv(hand_color)
        glBegin(GL_POINTS)
        for point in palm_points:
            glVertex3fv(point)
        glEnd()
    
    def draw_fingers(self, angles):
        finger_color = (0.6, 1.0, 0.6)  # Adjust color if needed
        glColor3fv(finger_color)

        # Define finger joints and lengths (modify as needed)
        finger_joints = [i for i in range(4)]  # Example: Five joints in each finger
        pinky_finger_lengths = [(0.2, 0.2, 0.2)]  # Example: Length of each finger segment
        finger_lengths = pinky_finger_lengths + 3*[(0.3,0.2,0.2)] # Example: Length of each finger segment

        for finger in range(5):
            base_point = (0, 0, 0)  # Base point for each finger
            glPushMatrix()

            # Translate to the base point of the finger
            glTranslatef(base_point[0], base_point[1], base_point[2])

            for joint in finger_joints:
                # Calculate the angle and distance for the finger segment
                segment_angles = angles[joint + finger * len(finger_joints): (joint + 1) + finger * len(finger_joints)]
                segment_length = finger_lengths[joint % len(finger_lengths)]

                # Calculate the endpoint of the finger segment
                endpoint = self.calculate_point(base_point, segment_angles, [segment_length])

                # Draw line from base to endpoint of the finger segment
                glBegin(GL_LINES)
                glVertex3fv(base_point)
                glVertex3fv(endpoint)
                glEnd()

                # Set the base point for the next segment as the current endpoint
                base_point = endpoint

            glPopMatrix()

    def draw_hand(self, angles):

        self.draw_palm()
        # self.draw_fingers(angles)

    def draw_hand_cylinder(self, angles):
        hand_color = (1.0, 0.8, 0.6)
        glColor3fv(hand_color)

        # Define finger connections (modify as needed)
        finger_connections = [(0, 1), (1, 2), (2, 3), (3, 4)]

        for finger in range(5):
            for connection in finger_connections:
                start = connection[0] + finger * 4
                end = connection[1] + finger * 4

                glPushMatrix()
                glTranslatef(0.0, 0.0, 0.0)  # Set the initial position of the cylinder

                # Calculate the start and end points for the finger connection
                start_point = self.calculate_point(self.origin, angles[start:end + 1], [1.0] * len(connection))
                end_point = self.calculate_point(self.origin, angles[end:end + 1], [1.0])

                # Calculate the distance between start and end points for the cylinder height
                distance = ((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2 + (end_point[2] - start_point[2])**2) ** 0.5

                # Calculate the direction vector for the cylinder orientation
                direction = (
                    end_point[0] - start_point[0],
                    end_point[1] - start_point[1],
                    end_point[2] - start_point[2]
                )

                # Calculate the angle for rotation around the direction vector
                angle = math.acos(direction[2] / distance) * 180.0 / math.pi
                axis = (
                    -direction[1] * direction[2],
                    direction[0] * direction[2],
                    0
                )

                glTranslatef(start_point[0], start_point[1], start_point[2])
                glRotatef(angle, *axis)
                gluCylinder(gluNewQuadric(), 0.1, 0.1, distance, 10, 10)  # Adjust radius and resolution as needed

                glPopMatrix()

    def draw_hand_test(self, angles):
        hand_color = (1.0, 0.8, 0.6)
        glLineWidth(2)
        glBegin(GL_LINES)

        # Define finger connections (modify as needed)
        finger_connections = [(0, 1), (1, 2), (2, 3), (3, 4)]

        for finger in range(5):
            for connection in finger_connections:
                start = connection[0] + finger * 4
                end = connection[1] + finger * 4
                glColor3fv(hand_color)
                glVertex3fv(self.calculate_point((0, 0, 0), angles[start:end + 1], [0.0] * len(connection)))

        glEnd()

    def run(self):
        pygame.init()
        display = (800, 600)
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
        gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
        glTranslatef(0.0, 0.0, -5)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.draw_hand(self.angle_frames[0])  # Display the hand using the first set of angles
            pygame.display.flip()
            pygame.time.wait(10)  # Adjust for frame rate

    def run_test(self):
        pygame.init()
        display = (800, 600)
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
        gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
        glTranslatef(0.0, 0.0, -5)

        frame_index = 0
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            glRotatef(1, 3, 1, 1)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.draw_hand(self.angle_frames[frame_index % len(self.angle_frames)])
            pygame.display.flip()
            pygame.time.wait(10)  # Adjust for frame rate
            frame_index += 1

if __name__ == "__main__":
    hand_model_3d = HandModel3D('angles.txt')  # Replace 'angles.txt' with your file name
    hand_model_3d.run()


