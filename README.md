# Optimized-UV-Disinfection

/*

* Copyright 2021 University of Illinois Board of Trustees. All Rights Reserved.

* Licensed under the “Non-exclusive Research Use License” (the "License");

* The License is included in the distribution as LICENSE.txt file.

* See the License for the specific language governing permissions and imitations under the License.

*/

This is the public repository for the code of the paper "Optimized Coverage Planning for UV Surface Disinfection", ICRA 2021

TODO: ADD Instructions on how to run the code

# Getting this repository up and running:

You will want to install the python libraries contained in requirements.txt (pip install -r requirements.txt)

This specific implentation depends on two pieces of software under separate liceses: Gurobi and LKH-3.0.6

To get gurobi up and running on your system,you can follow the instructions here (for academic licenses) :       https://www.gurobi.com/academia/academic-program-and-licenses/

To install LKH-3.0.6 you can follow the instructions on the project's website: http://webhotel4.ruc.dk/~keld/research/LKH-3/

## IMPORTANT: After installing LKH make sure to change lines 24-26 of the file tspSolver5.py to reflect the appropriate installation path

