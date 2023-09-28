# Driving Simulator

This repository contains the project done as a part of CS747: Foundations of Intelligent and Learning Agents at IIT Bombay in Autumn 2022.

## Problem Statement

This project aims to design a controller to drive a car using Reinforcement Learning and help the car navigate to its final destination while avoiding several obstacles on its way. This project considers two scenarios :- 

1. The car is present in a parking lot with no obstacles present in the parking lot. The objective of the controller is to exit the parking lot successfully in given number of time steps.

2. The car is present in another parking lot filled with pools of mud. The objective of the controller is to avoid these obstacles and exit the parking lot in given number of time steps.

## Scenario 1 : Clean Parking Lot

The controller will be navigating a 50 x 25 car out of a 700 x 700 square parking lot. It must exit onto the road whose entrance has its centre located at (350, 0). The controller must successfully navigate the car out of the parking lot in 1000 time steps. The car will be initialised at a random position and orientation in the parking lot. The task is deemed unsuccessful if the car bumps into a wall of the parking lot or if it is not able to exit the parking lot within 1000 time steps.

<img width="361" alt="clean_parking_lot" src="https://github.com/Adu3108/DrivingSimulator/assets/81511060/6e386c11-d2fd-4786-9e65-801009689899">

## Scenario 2 : Dirty Parking Lot

Just like the first scenario, the controller will be navigating a 50 x 25 car out of a 700 x 700 square parking lot. However, this time the exit to the parking lot will be narrower. Additionally, the parking lot will be filled with 4 randomnly located pits of mud each of size 100 x 100. Other rules apply just like before.

<img width="362" alt="dirty_parking_lot" src="https://github.com/Adu3108/DrivingSimulator/assets/81511060/c49f01cb-bf8b-44f3-b014-a1dedec48c0a">
