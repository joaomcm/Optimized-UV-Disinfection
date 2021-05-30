import pandas as pd
from tqdm import tqdm

from planning.disinfection_3D import Disinfection_Problem
from planning.robot_cspaces import Robot_3D_CSpace,CSpaceObstacleSolver 

problem = Disinfection_Problem(
    total_dofs = 11,
    linear_dofs = [0,1],
    angular_dofs = [4,5,6,7,8],
    frozen_dofs = [2,9],
    base_height_link = 2,
    robot_height = 1.5,
    lamp_linknum = 11,
    lamp_local_coords = [0,0,0],
    active_dofs = [0,1,4,5,6,7,8,9],
    robot_cspace_generator = Robot_3D_CSpace,
    robot_cspace_solver = CSpaceObstacleSolver,
    float_height = 0.08
)
experiment = '30_min'
tmax = 0.5
total_distances = []
coverages = []
resolutions = []

#resolution here is in mm
for res in tqdm([750]):#,20,30,40,50,60,70,80]):
    total_distance,coverage,resolution,res_dir = problem.perform_experiment(
    results_dir = './3D_results',
    mesh_file = './data/environment_meshes/remeshed_hospital_room_full_35k.obj',
    min_distance = 0.05,
    from_scratch = True,
    irradiance_from_scratch = True,
    float_height = 0.08,
    power = 80,
    resolution = res,
    experiment = experiment,
    tmax = tmax,
    robot_name = 'armbot')
    total_distances.append(total_distance)
    coverages.append(coverage)
    resolutions.append(resolution)
    df = pd.DataFrame({'resolution':resolutions,'coverage':coverages,'total_distance':total_distances})
    df.to_csv(res_dir + '/3D_armbot_results.csv', sep = '|', index = False)
