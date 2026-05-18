import cv2
import numpy as np
import os
import pandas as pd

# # --- Original Interactive Tool (Commented Out) ---
# # Read image
# img_path = r'E:\StudyMaterial\France\DeepLearning\Group_Project\relighting\image\test.jpg'
# img = cv2.imread(img_path)
# 
# if img is None:
#     print("Error: Unable to read image")
# else:
#     # Set scaling factor (0.2 means shrinking to 20%)
#     scale_percent = 0.5 
#     width = int(img.shape[1] * scale_percent)
#     height = int(img.shape[0] * scale_percent)
#     dim = (width, height)
#     
#     # Resize image for previewing
#     resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
# 
#     def click_event(event, x, y, flags, params):
#         if event == cv2.EVENT_LBUTTONDOWN:
#             # Scale the coordinates from the preview back to the original high-res image
#             real_x = int(x / scale_percent)
#             real_y = int(y / scale_percent)
#             print(f"--- Original Image Coordinates: FIXED_X = {real_x}, FIXED_Y = {real_y} ---")
#             
#             # Draw a point on the preview image to visually confirm
#             cv2.circle(resized, (x, y), 5, (0, 0, 255), -1)
#             cv2.imshow('Click Sphere Center', resized)
# 
#     cv2.imshow('Click Sphere Center', resized)
#     cv2.setMouseCallback('Click Sphere Center', click_event)
#     print("Please click the center of the chrome sphere in the popup window...")
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# --- 1. Use my precisely measured coordinates ---
FIXED_X = 531
FIXED_Y = 847
FIXED_R = 30  # The green circle needs to accurately align with the sphere's edge

# --- 2. Path Configurations ---
input_folder = r'E:\StudyMaterial\France\DeepLearning\Group_Project\relighting\image'
output_folder = r'E:\StudyMaterial\France\DeepLearning\Group_Project\relighting\image_markSphere'
excel_output = r'E:\StudyMaterial\France\DeepLearning\Group_Project\relighting\light_directions.xlsx'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def batch_process_fixed_fixed():
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    # Used to store row data, structured as: [Image_Name, L_x, L_y, L_z]
    data_list = []

    print(f"Starting batch processing for {len(image_files)} high-resolution images...")

    for filename in image_files:
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        if img is None: 
            continue

        # Use floating-point numbers for computation
        x_c, y_c, r = float(FIXED_X), float(FIXED_Y), float(FIXED_R)

        # --- 3. Find the highlight (Specularity) in the Red Channel ---
        gray_red = img[:, :, 2]
        mask = np.zeros_like(gray_red)
        cv2.circle(mask, (int(x_c), int(y_c)), int(r * 0.9), 255, -1)
        search_area = cv2.bitwise_and(gray_red, mask)
        
        _, _, _, max_loc = cv2.minMaxLoc(search_area)
        x_h, y_h = float(max_loc[0]), float(max_loc[1])

        # ==================== 4. Compute 3D Vector (Strictly mapped to PBRT Coordinate System) ====================
        dx = x_h - x_c
        dy = y_h - y_c
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist > r:
            dx = (dx / dist) * r
            dy = (dy / dist) * r
            dist = r

        # 1. Map to the image projection space
        nx_img = dx / r
        ny_img = dy / r
        
        # 2. Map to the normal vector N in the PBRT physical coordinate system
        # X-axis: Right is positive -> nx_pbrt = nx_img
        # Z-axis: Up is positive. In image space, dy is negative up and positive down -> nz_pbrt = -ny_img
        nx_pbrt = nx_img
        nz_pbrt = -ny_img 
        
        # Y-axis (Depth): Since it faces the camera hemisphere, the normal vector points out of the screen (negative Y direction)
        ny_pbrt_sq = 1.0 - nx_pbrt**2 - nz_pbrt**2
        ny_pbrt = -np.sqrt(max(0, ny_pbrt_sq))  # Facing out of the screen
        
        N_pbrt = np.array([nx_pbrt, ny_pbrt, nz_pbrt])

        # 3. Strictly compute the light source vector L based on the law of reflection
        # The view vector V must point from the surface toward the camera (i.e., out of the screen, negative Y direction)
        V_pbrt = np.array([0, -1, 0]) 
        
        # Standard specular reflection formula: L = 2 * (N · V) * N - V
        dot_nv = np.dot(N_pbrt, V_pbrt)
        
        L = 2.0 * dot_nv * N_pbrt - V_pbrt
        
        # Normalize the light source vector
        L = L / np.linalg.norm(L)
        
        # Save to the list: [Filename, X component, Y component, Z component]
        data_list.append([filename, L[0], L[1], L[2]])
        # =========================================================================

        # --- 5. Annotate and Save Debug Images ---
        debug_img = img.copy()
        cv2.circle(debug_img, (int(x_c), int(y_c)), int(r), (0, 255, 0), 2)
        cv2.circle(debug_img, (int(x_h), int(y_h)), 5, (0, 0, 255), -1)
        cv2.imwrite(os.path.join(output_folder, f"mark_{filename}"), debug_img)
        print(f"Processed {filename} -> physical L: [{L[0]:.4f}, {L[1]:.4f}, {L[2]:.4f}]")

    # ==================== 6. Create DataFrame and Save to Excel ====================
    if data_list:
        # Define the four column names for the Excel file
        columns = ['Image_Name', 'Light_Dir_X', 'Light_Dir_Y', 'Light_Dir_Z']
        df = pd.DataFrame(data_list, columns=columns)
        
        # Export to Excel. index=False means Pandas default numeric row indexing will not be saved
        df.to_excel(excel_output, index=False)
        print("-" * 30)
        print(f"Success! Light vectors successfully saved to Excel: {excel_output}")
    else:
        print("No images were processed; no Excel file was generated.")

if __name__ == "__main__":
    batch_process_fixed_fixed()