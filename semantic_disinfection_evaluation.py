import pandas as pd
from visibility import Visibility
from glob import glob
import trimesh as tm
import pickle
import os
import numpy as np
from scipy.sparse import lil_matrix
from tqdm import tqdm

def get_irradiance_matrix(vis_tester,sampling_places):
    total_faces = vis_tester.mesh.mesh.faces.shape[0]
    irradiance_matrix = lil_matrix((sampling_places.shape[0],total_faces))
    for i in range(sampling_places.shape[0]):
        _,irradiance = vis_tester.render(id0 =None,id1 = None,pos = sampling_places[i,:].tolist())
        irradiance = irradiance
        irradiance_matrix[i,np.where(irradiance > 0)] = irradiance[irradiance > 0]

    return irradiance_matrix.tocsr()

def evaluate_semantic_disinfection_performance(roadmap):
    mesh_name = roadmap.split('/')[-4]
    # we then load the mesh original mesh and its ground truth:
    mesh_file = '/home/motion//Optimized-UV-Disinfection/data/environment_meshes/aug_10_entire_val_ade20kmodel_vanilla_prob_weighting/estimated_scannet_val/{}.ply'.format(mesh_name)
    mesh_gt_file = '/home/motion//Optimized-UV-Disinfection/data/environment_meshes/aug_10_entire_val_ade20kmodel_vanilla_prob_weighting/estimated_scannet_val/gt_{}.ply'.format(mesh_name)
    ground_truth_visibility = Visibility(mesh_gt_file,res = 512, useShader = True,createWnd = True)
    main_dir = os.path.dirname(roadmap)
    reachable_file = os.path.join(main_dir,'armbot_reachable_330_divs.p')
    solutions_file = os.path.join(main_dir,'armbot_solutions_330_divs.p')
    sampling_places_file = os.path.join(main_dir,'armbot_sampling_places_330_divs.p')

    reachable = pickle.load(open(reachable_file,'rb'))
    solutions = np.array(pickle.load(open(solutions_file,'rb'))[0])
    sampling_places = pickle.load(open(sampling_places_file,'rb'))

    selected_points = sampling_places[reachable,:][solutions>0,:]
    irradiance_matrix = get_irradiance_matrix(ground_truth_visibility,selected_points)

    gt_mesh = ground_truth_visibility.mesh.mesh
    final_irradiances = 80*irradiance_matrix.transpose()@solutions[solutions>0]
    areas = gt_mesh.area_faces
    colors = gt_mesh.visual.face_colors
    label = colors[:,0] >= 255
    disinfected = final_irradiances > 280
    total_ht_area = np.sum(areas[label])
    total_area = np.sum(areas)
    disinfected_ht_area = np.sum(areas[label][disinfected[label]])
    disinfected_area = np.sum(areas[disinfected])
    ht_area_fraction = total_ht_area/total_area
    return mesh_name,total_area,ht_area_fraction,disinfected_ht_area/total_ht_area,disinfected_area/total_area



sa_roadmap = sorted(glob('/home/motion/Optimized-UV-Disinfection/3D_results/Semantic/*/surface_agnostic/armbot/armbot_roadmap_330_divs.p'))
# sa_sampling_places = glob('/home/motion/Optimized-UV-Disinfection/3D_results/Semantic/*/surface_agnostic/armbot/armbot_sampling_places_330_divs.p')


sst_roadmap = sorted(glob('/home/motion/Optimized-UV-Disinfection/3D_results/Semantic/*/soft_semantic_thresholding/armbot/armbot_roadmap_330_divs.p'))
# sst_sampling_places = glob('/home/motion/Optimized-UV-Disinfection/3D_results/Semantic/*/soft_sematic_thresholding/armbot/armbot_sampling_places_330_divs.p')


# result = evaluate_semantic_disinfection_performance(sst_roadmap[0])

roadmaps = sorted(glob('/home/motion/Optimized-UV-Disinfection/3D_results/Semantic/*/*/armbot/armbot_roadmap_330_divs.p'))


mesh_names = []
experiments = []
durations = []
total_areas = []
ht_area_fractions = []
ht_disinfected_fraction = []
total_disinfected_fraction = []
old_mesh = ''
for rm in tqdm(roadmaps):
    experiment = rm.split('/')[-3]
    duration = experiment.split('_')[-2]
    experiments.append(experiment)
    durations.append(duration)
    results = evaluate_semantic_disinfection_performance(rm)
    mesh_names.append(results[0])
    total_areas.append(results[1])
    ht_area_fractions.append(results[2])
    ht_disinfected_fraction.append(results[3])
    total_disinfected_fraction.append(results[4])
results_df = pd.DataFrame({'mesh_name':mesh_names,'experiment':experiments,'duration':durations,'room area':total_areas,
                          'ht_area_fraction':ht_area_fractions,'ht_disinfected_fraction':ht_disinfected_fraction,'total_disinfected_fraction':total_disinfected_fraction})

results_df.to_csv('./Semantic_results_shorter_time.csv',sep = '|')