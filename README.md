# IRP-REAL-TIME-CAR-AVODIANCE AND LANE-FOLLOWING

## Project Overview

This project focuses on the development of a real-time lane detection and obstacle avoidance framework implemented on the **Quanser QCar autonomous vehicle platform**. The system integrates vision-based perception (using an RGB camera for lane detection) with LiDAR-based obstacle sensing, and a Finite State Machine (FSM) for intelligent decision-making. The goal is to enable the QCar to autonomously navigate a predefined path while safely avoiding obstacles in real-time within structured indoor environments.

## Features

*   **Real-time Lane Detection:** Utilizes an RGB camera and image processing techniques (color segmentation in HSV color space) to detect yellow lane markings and estimate the vehicle's lateral position for stable lane following.
*   **LiDAR-based Obstacle Detection:** Employs a 360° LiDAR sensor to provide accurate distance and positional information, enabling reliable detection of obstacles within a predefined safety range.
*   **Finite State Machine (FSM) Control:** Implements a structured FSM with states such as "Lane Following," "Obstacle Detection," "Lateral Shift," "Forward Bypass," and "Lane Rejoining" to manage vehicle behavior and ensure smooth, predictable avoidance maneuvers.
*   **Sensor Fusion:** Combines data from both vision and LiDAR sensors for enhanced environmental awareness and robust performance.
*   **Python-based Software Framework:** The system is implemented using Python and interfaces with the QCar hardware via the QCar SDK.

## Hardware Used

*   **Quanser QCar Platform:** A compact, research-oriented autonomous vehicle.
*   **RGB Camera:** Mounted at the front for vision-based lane detection.
*   **360° LiDAR Sensor:** Provides continuous range measurements for obstacle detection.
*   **NVIDIA Jetson TX2:** Onboard computation for real-time processing.

## Software and Libraries

*   **Python:** Primary programming language.
*   **OpenCV:** Used for image processing in lane detection.
*   **Quanser QCar SDK:** (`pal.products.qcar` library) for hardware interfacing.
*   **NumPy:** For numerical operations.
*   **json, time:** Standard Python libraries.

## Setup and Running

The project was developed for and tested on a physical Quanser QCar. While specific setup and execution instructions are typically provided with the code, the core logic would involve:

1.  **Environment Setup:** Ensure a Python environment is configured with the necessary libraries (OpenCV, NumPy, Quanser SDK).
2.  **QCar Connection:** Establish a connection to the Quanser QCar.
3.  **Code Execution:** Run the main Python script (`qcar_main.py` based on the provided `CODE.docx`) on the QCar's onboard computer.

*Note: The actual code and its execution details are expected to be found in `CODE.docx` and would need to be adapted based on the specific Quanser QCar setup.*

## Results

Experimental validation on an indoor test track demonstrated:
*   Reliable lane tracking with minimal oscillations.
*   Accurate obstacle detection.
*   Over 90% success rate in obstacle avoidance during indoor tests.
*   Smooth and consistent avoidance behavior with minimal latency.





## Team

*   **Arihant A** (01FE22BEI048)
*   **Shivani D** (01FE22BEC293)
*   **Radhika M** (01FE22BEC296)

**Under the Guidance of:** Dr. Basawaraj
**Institution:** KLE Technological University
**Semester:** VII, 2024-2025
