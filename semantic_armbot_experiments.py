import pandas as pd
from tqdm import tqdm
from glob import glob
from planning.disinfection3d import DisinfectionProblem
from planning.robot_cspaces import Robot3DCSpace,CSpaceObstacleSolver,UnreachablePointsError

problem = DisinfectionProblem(
    total_dofs = 11,
    linear_dofs = [0,1],
    angular_dofs = [4,5,6,7,8],
    frozen_dofs = [2,9],
    base_height_link = 2,
    robot_height = 1.5,
    lamp_linknum = 11,
    lamp_local_coords = [0,0,0],
    active_dofs = [0,1,4,5,6,7,8,9],
    robot_cspace_generator = Robot3DCSpace,
    robot_cspace_solver = CSpaceObstacleSolver,
    float_height = 0.08,
    initial_points = 500
)
experiment = 'surface_agnostic'
tmax = 0.5
total_distances = []
coverages = []
resolutions = []

all_meshes = glob('/home/motion//Optimized-UV-Disinfection/data/environment_meshes/aug_10_entire_val_ade20kmodel_vanilla_prob_weighting/estimated_scannet_val/*.ply')

meshes_series = pd.Series(all_meshes)
estimated_mask = ~meshes_series.str.split('/',expand = True).iloc[:,-1].str.startswith('gt')
# we now get the number of done meshes

# done_files = glob('/home/motion/Optimized-UV-Disinfection/3D_results/Semantic/*/{}/armbot/*'.format(experiment))
# if(done_files):
#     done_meshes = pd.Series(done_files).str.split('/',expand = True).iloc[:,-4].tolist()
# else:
#     done_meshes = []

estimated_meshes = sorted(meshes_series[estimated_mask].tolist())
failed_meshes = []
experiments = ['surface_agnostic_{}_minutes','soft_thresholding_{}_minutes','hard_cutoff_25_{}_minutes','hard_cutoff_50_{}_minutes']
# experiments = ['hard_cutoff_20_{}_minutes']
for experiment in experiments: 
    if(experiment == 'surface_agnostic_{}_minutes'):
        hard_cutoff = False
        cutoff_threshold = 0
    elif(experiment == 'soft_thresholding_{}_minutes'):
        hard_cutoff = False
        cutoff_threshold = 0.5
    else:
        hard_cutoff = True
        cutoff_threshold = float(experiment.split('_')[2])/100
    for time_limit in [2,4,6,8,10,30,6000]:
        tmax = time_limit/60
        this_experiment = experiment.format(time_limit)
        for mesh_file in estimated_meshes[:5:2]:
            try:
                res = 330
                mesh_name = mesh_file.split('/')[-1].split('.')[0]
                print(mesh_name)
                # if(mesh_name not in done_meshes):
                print('\n\n\n\n performing experiment {} on mesh file = {} \n\n\n'.format(this_experiment, mesh_file))
                total_distance,coverage,resolution,res_dir = problem.perform_experiment(
                results_dir = './3D_results/Semantic',
                mesh_file = mesh_file,
                min_distance = 0.05,
                from_scratch = True,
                irradiance_from_scratch = True,
                float_height = 0.15,
                power = 80,
                resolution = res,
                experiment = this_experiment,
                tmax = tmax,
                robot_name = 'armbot',
                hard_cutoff = hard_cutoff,
                cutoff_threshold = cutoff_threshold)
            except Exception as e:
                print('initial planning failed for mesh {}!'.format(mesh_name))
                with open('./failed_meshes.txt','a') as f:
                    f.write(mesh_name+'\r\n')
                pass
            
    # else:
    #     # print('\n\n\n\n\n\n\n mesh already done - skipping mesh {}\n\n\n\n\n\n\n\n\n'.format(mesh_file))
    #     pass