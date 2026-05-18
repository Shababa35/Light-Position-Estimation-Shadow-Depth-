import os
import subprocess
import numpy as np
import csv

# --- Configuration ---
NUM_IMAGES = 1 
PBRT_EXECUTABLE = r"C:\Users\Jiachen\pbrt\Release\pbrt.exe"
BASE_DIR = os.path.abspath("monkey_random_dataset")
RENDER_DIR = os.path.join(BASE_DIR, "renders")
SCENE_DIR = os.path.join(BASE_DIR, "scenes")
MONKEY_FILE = os.path.abspath("ex01_blender_monkey.pbrt")

for folder in [RENDER_DIR, SCENE_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def generate_random_light():
    phi = np.random.uniform(0, 2 * np.pi)
    theta = np.random.uniform(0, np.radians(75)) 
    r = 100.0 
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta) 
    return x, y, z

def generate_random_position():
    tx = np.random.uniform(-3.0, 3.0)
    ty = np.random.uniform(-3.0, 3.0)
    tz = np.random.uniform(0.0, 1.5) # Reduced max height slightly for better framing
    return tx, ty, tz

def generate_random_rotation():
    # Returns a random angle between 0 and 360 degrees
    return np.random.uniform(0, 360)

gt_log = []

for i in range(NUM_IMAGES):
    lx, ly, lz = generate_random_light()
    mx, my, mz = generate_random_position()
    m_rot = generate_random_rotation() # New random rotation
    
    img_name = f"monkey_{i:04d}.png"
    output_img_path = os.path.join(RENDER_DIR, img_name).replace("\\", "/")
    temp_pbrt_path = os.path.join(SCENE_DIR, f"scene_{i:04d}.pbrt")
    formatted_monkey_path = MONKEY_FILE.replace("\\", "/")

    pbrt_content = f"""
LookAt 0 15 10  0 0 0  0 0 1
Camera "perspective" "float fov" 50
Sampler "halton" "integer pixelsamples" 64
Integrator "path"
Film "rgb" "string filename" "{output_img_path}"
     "integer xresolution" 256 "integer yresolution" 256

WorldBegin
LightSource "infinite" "rgb L" [ .1 .1 .1 ]
LightSource "distant" "point3 from" [ {lx} {ly} {lz} ] "point3 to" [ 0 0 0 ] "blackbody L" 5000 "float scale" 1.5

AttributeBegin
  Translate {mx} {my} {mz}
  Rotate {m_rot} 0 0 1
  Scale 1.5 1.5 1.5 
  Include "{formatted_monkey_path}"
AttributeEnd

AttributeBegin
  Material "diffuse" "rgb reflectance" [0.4 0.1 0.1]
  Translate 0 0 -0.5
  Shape "trianglemesh" "integer indices" [0 1 2 0 2 3]
        "point3 P" [ -15 -15 0  15 -15 0  15 15 0  -15 15 0 ]
AttributeEnd
"""

    with open(temp_pbrt_path, 'w') as f:
        f.write(pbrt_content)

    print(f"Rendering {img_name} (Rot: {m_rot:.1f}°) ...")
    result = subprocess.run([PBRT_EXECUTABLE, temp_pbrt_path], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        break 
    
    l_vec = np.array([lx, ly, lz])
    l_vec /= np.linalg.norm(l_vec)
    
    # Updated ground truth to include m_rot
    gt_log.append([img_name, l_vec[0], l_vec[1], l_vec[2], mx, my, mz, m_rot])

# Save labels with rotation column
if gt_log:
    with open(os.path.join(BASE_DIR, "labels.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "light_x", "light_y", "light_z", "monkey_x", "monkey_y", "monkey_z", "rotation_z"])
        writer.writerows(gt_log)
    print(f"\nDone! Dataset with random rotation generated in {BASE_DIR}")