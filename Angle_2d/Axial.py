import os
import nibabel as nib
import numpy as np
import csv
from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import euclidean
from math import acos, degrees

def load_nifti(file_path):
    nifti = nib.load(file_path)
    return nifti

def find_boundary_points(region_mask, tumor_mask):
    boundary_points = []
    for x in range(1, region_mask.shape[0] - 1):
        for y in range(1, region_mask.shape[1] - 1):
            if region_mask[x, y] > 0 and tumor_mask[x, y] > 0:
                neighbors = region_mask[x-1:x+2, y-1:y+2]
                if np.any(neighbors == 0):
                    boundary_points.append((x, y))
    return boundary_points

def analyze_region_relationship_and_centroid(blood_vessel_data, tumor_data, intensity):
    max_wrap_angle = 0
    max_wrap_details = ""
    wrap_status = "无接触"
    has_light_contact = False
    
    for i in range(blood_vessel_data.shape[2]):
       # print(f"Analyzing layer {i+1}")
        layer_bv = blood_vessel_data[:, :, i]
        layer_tumor = tumor_data[:, :, i]
        
        if not np.any(np.isclose(layer_bv, intensity)):
           # print(f"No blood vessel data with intensity close to {intensity} in layer {i+1}")
            continue

        labeled_layer, num_features = label(np.isclose(layer_bv, intensity))

        for region in range(1, num_features + 1):
            region_mask = labeled_layer == region
            region_area = np.sum(region_mask)
            intersection_mask = region_mask & (layer_tumor > 0)
            intersection_area = np.sum(intersection_mask)
            non_overlap_area = region_area - intersection_area
            
            if intersection_area == region_area:
                #print(f"第{i+1}层, 区域{region}：完全包绕")
                wrap_status = "完全包绕"
                return wrap_status
            
            elif intersection_area > 0:
                if non_overlap_area <= 2:
                    #print(f"第{i+1}层, 区域{region}：部分包绕，但仅{non_overlap_area}个像素未包绕，无法计算角度")
                    has_light_contact = True
                    continue

                centroid = center_of_mass(region_mask)
                centroid_in_intersection = intersection_mask[int(centroid[0]), int(centroid[1])]
                
                boundary_points = find_boundary_points(region_mask, layer_tumor)
                
                if len(boundary_points) < 2:
                    #print(f"第{i+1}层, 区域{region}：部分包绕，但找不到足够的交点，仅{len(boundary_points)}个像素接触，无法计算角度")
                    has_light_contact = True
                    continue

                point_b, point_c = boundary_points[0], boundary_points[-1]
                
                if point_b == point_c:
                    #print(f"第{i+1}层, 区域{region}：交点b和c是相同的点，忽略计算")
                    has_light_contact = True
                    continue

                distance_ab = euclidean(centroid, point_b)
                distance_ac = euclidean(centroid, point_c)
                distance_bc = euclidean(point_b, point_c)

                try:
                    raw_angle_bac = degrees(acos((distance_ab**2 + distance_ac**2 - distance_bc**2) / (2 * distance_ab * distance_ac)))
                    if centroid_in_intersection:
                        wrap_angle = 360 - raw_angle_bac
                    else:
                        wrap_angle = raw_angle_bac
                except ValueError:
                    wrap_angle = None  

                if wrap_angle and wrap_angle > max_wrap_angle:
                    max_wrap_angle = wrap_angle
                    max_wrap_details = (
                        f"部分包绕, 包绕角度: {wrap_angle:.2f}度"
                    )
                    wrap_status = "部分包绕"

    if wrap_status == "部分包绕":
        return max_wrap_details
    elif has_light_contact:
        return "轻微接触 无包绕"
    else:
        return wrap_status

def analyze_patient_blood_vessels(blood_vessel_path, tumor_path):
    intensities = [1, 2, 3]  

    blood_vessel_data = load_nifti(blood_vessel_path).get_fdata()
    tumor_data = load_nifti(tumor_path).get_fdata()

    blood_vessel_data = np.rint(blood_vessel_data).astype(np.float32)
    tumor_data = np.rint(tumor_data).astype(np.float32)

    results = {}
    for intensity in intensities:
        result = analyze_region_relationship_and_centroid(blood_vessel_data, tumor_data, intensity)
        results[intensity] = result
        print(f"Result for intensity {intensity}: {result}")
    
    return results

def analyze_all_patients(blood_vessel_dir, tumor_dir, output_csv):
    blood_vessel_files = sorted(os.listdir(blood_vessel_dir))
    tumor_files = sorted(os.listdir(tumor_dir))

    assert len(blood_vessel_files) == len(tumor_files), "The number of blood vessel and tumor mask files must be the same"
    
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Patient ID', 'Intensity 1', 'Intensity 2', 'Intensity 3'])
        
        for blood_vessel_file, tumor_file in zip(blood_vessel_files, tumor_files):
            patient_id = os.path.splitext(os.path.splitext(blood_vessel_file)[0])[0]
            blood_vessel_path = os.path.join(blood_vessel_dir, blood_vessel_file)
            tumor_path = os.path.join(tumor_dir, tumor_file)
            
            print(f"Processing patient: {blood_vessel_file}")
            patient_results = analyze_patient_blood_vessels(blood_vessel_path, tumor_path)
            
            writer.writerow([patient_id] + [patient_results.get(intensity, '') for intensity in [1, 2, 3]])

blood_vessel_dir = '#血管mask'
tumor_dir = '#肿瘤mask'
output_csv = '#输出WrapCalc_2D.csv'

analyze_all_patients(blood_vessel_dir, tumor_dir, output_csv)
