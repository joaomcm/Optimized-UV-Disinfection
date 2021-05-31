# Optimized-UV-Disinfection

Copyright 2021 University of Illinois Board of Trustees. All Rights Reserved.

Licensed under the “Non-exclusive Research Use License” (the "License");

The License is included in the distribution as [LICENSE.txt](https://github.com/joaomcm/Optimized-UV-Disinfection/blob/main/LICENSE.txt) file.

See the License for the specific language governing permissions and imitations under the License.


# About

This is the public repository for the code of the paper "Optimized Coverage Planning for UV Surface Disinfection", ICRA 2021.

Authors: Joao Correia Marques, Ramya Ramalingam, Zherong Pan, and Kris Hauser

Contact: [Joao Correia Marques](mailto:jmc12@illinois.edu)

Requires Python 3.x (3.7+ recommended) and an OpenGL 4.1-compatible graphics card for visibility calculations.

# Installing Dependencies

This program depends on two pieces of software distributed under separate liceses: Gurobi and LKH-3.0.6.  To install:

1. Install the Python libraries contained in requirements.txt (pip install -r requirements.txt)

2. To get gurobi up and running on your system,you can follow the instructions here (for academic licenses) :       https://www.gurobi.com/academia/academic-program-and-licenses/

3. To install LKH-3.0.6 you can follow the instructions on the project's website: http://webhotel4.ruc.dk/~keld/research/LKH-3/

4. After installing LKH place it in the `Optimized-UV-Disinfection/LKH-3.0.6` folder


# Reproducing our Results

In order to reproduce the results in the paper,  run `python {robot_model}_experiments.py`.

In order to visualize the results as a movie, the `{Robot Model} Animations.ipynb` Jupyter notebooks will create a series of still images that can be used to create a movie using ffmpeg or your program of choice. 


# Using our planner with different robots & environments

To use the planner with a different environment, you should be able to simply change the "mesh_file" argument to any of the `{robot}_experiments.py` file.

`Towerbot_experiments.py` and `floatbot_experiments.py` give examples of how to adopt our pipeline to different robots. The first is a mobile base robot that has no rotational joints and a light that is not a point light-source, thus showing how to adapt to more general light models.  

`floatbot` is a robot that floats in 3D space, as though it were a quadrobot carrying a light source. This example shows what changes need to be made in case of larger differences between tested robots and the default robot - armbot.

For further inquiries, please contact us. 
