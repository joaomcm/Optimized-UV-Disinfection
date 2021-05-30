# Optimized-UV-Disinfection

/*

* Copyright 2021 University of Illinois Board of Trustees. All Rights Reserved.

* Licensed under the “Non-exclusive Research Use License” (the "License");

* The License is included in the distribution as LICENSE.txt file.

* See the License for the specific language governing permissions and imitations under the License.

*/

This is the public repository for the code of the paper "Optimized Coverage Planning for UV Surface Disinfection", ICRA 2021


# Getting this repository up and running:

You will want to install the python libraries contained in requirements.txt (pip install -r requirements.txt)

This specific implentation depends on two pieces of software under separate liceses: Gurobi and LKH-3.0.6

To get gurobi up and running on your system,you can follow the instructions here (for academic licenses) :       https://www.gurobi.com/academia/academic-program-and-licenses/

To install LKH-3.0.6 you can follow the instructions on the project's website: http://webhotel4.ruc.dk/~keld/research/LKH-3/

## IMPORTANT: After installing LKH make sure to place it on the LKH-3.0.6 folder! 

# Reproducing our Results

In order to reproduce our results you should be able to just run {robot_model}_experiments.py. In order to visualize the results as a movie, the {Robot Model} Animations jupyter notebooks should create a series of still images that can be used to create a movie using ffmpeg or your program of choice. 

# Using our planner with different robots & environments

To use the planner with a different environment, you should be able to simply change the "mesh_file" argument on either {robot}_experiments.
Towerbot_experiments.py and floatbot_experiments.py give examples of how to adopt our pipeline to different robots. The first deals with a somewhat similar robot that has no rotational joints and a light that is not a point light-source (thus showing how to adapt to more general light models) and the second deals with a robot that floats in 3D space, though showing what changes need to be made in case of larger differences between tested robots and the default robot - armbot.

For further inquiries, do not hesitate to contact us. We are happy to help.
