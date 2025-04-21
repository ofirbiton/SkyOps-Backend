#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Manual test file for sending inputs to the server with manual point selection.
Inputs:
  - Buildings.png file (buildings image)
  - Satelite.png file (satellite image)
  - The user manually selects 2 points from the satellite image:
       The first point is used as top_left_coord (takeoff, marked in green)
       The second point is used as bottom_right_coord (landing, marked in red)
       
After selecting the points, the Buildings image is updated by setting the corresponding pixels 
(with a single pixel mark) and saved as "Buildings_marked.png". The updated image is then shown 
for approval. After pressing any key, the images and coordinate strings are sent to the server 
(POST URL http://127.0.0.1:5000/api/create-mission).
"""

import cv2
import requests
import os

# File paths – update according to your file locations
BUILDINGS_PATH = r"C:\Users\ofirb\OneDrive\Desktop\Buildings.png"
SATELLITE_PATH  = r"C:\Users\ofirb\OneDrive\Desktop\Satelite.png"
MARKED_BUILDINGS_PATH = r"Buildings_marked.png"

def select_points(image_path):
    """
    Displays the satellite image in a window and allows the user to select 2 points by clicking.
    Returns the two points as tuples (x, y).
    """
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading the satellite image.")
        exit(1)
    
    points = []
    window_name = "Select 2 points in the satellite image (first: takeoff, second: landing)"
    
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 2:
                points.append((x, y))
                # Mark on the satellite image with a small circle (radius 3)
                color = (0, 255, 0) if len(points) == 1 else (0, 0, 255)
                cv2.circle(img, (x, y), 3, color, -1)
                cv2.imshow(window_name, img)
                print(f"Point selected: {(x, y)}")
            else:
                print("2 points have already been selected.")
    
    cv2.imshow(window_name, img)
    cv2.setMouseCallback(window_name, click_event)
    print("Click on the 'Satellite Image' window to select 2 points.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if len(points) < 2:
        print("Less than 2 points were selected. Exiting.")
        exit(1)
        
    return points[0], points[1]

def main():
    # API URL of the local server
    url = "http://127.0.0.1:5000/api/create-mission"
    
    # Select points from the satellite image (for visualization only; not used in conversion)
    print("Please select 2 points from the satellite image:")
    takeoff_point, landing_point = select_points(SATELLITE_PATH)
    
    # Instead of using the selected points for the diagonal,
    # we force the diagonal coordinates to be the fixed values:
    top_left_coord = "(197900, 591690)"
    bottom_right_coord = "(198509, 591263)"
    
    print("Fixed diagonal coordinates (for conversion):")
    print("Takeoff (top_left_coord):", top_left_coord)
    print("Landing (bottom_right_coord):", bottom_right_coord)
    
    # Update the Buildings image with the selected points (for marking purposes).
    buildings_img = cv2.imread(BUILDINGS_PATH)
    if buildings_img is None:
        print("Error loading the buildings image.")
        exit(1)
    
    # Mark the corresponding pixel in the Buildings image:
    # (Note: image indexing is [y, x])
    buildings_img[takeoff_point[1], takeoff_point[0]] = (0, 255, 0)   # Green for takeoff
    buildings_img[landing_point[1], landing_point[0]] = (0, 0, 255)     # Red for landing
    
    # Save the modified Buildings image
    cv2.imwrite(MARKED_BUILDINGS_PATH, buildings_img)
    print("Modified Buildings image saved as:", MARKED_BUILDINGS_PATH)
    
    # Show the modified Buildings image for approval.
    cv2.imshow("Marked Buildings Image - Approve (Press any key to continue)", buildings_img)
    print("Please check the marked Buildings image and press any key to approve.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Prepare the data for the POST request:
    files = {
        "buildings_image": open(MARKED_BUILDINGS_PATH, "rb"),
        "satellite_image": open(SATELLITE_PATH, "rb")
    }
    data = {
        "top_left_coord": top_left_coord,
        "bottom_right_coord": bottom_right_coord
    }
    
    try:
        response = requests.post(url, files=files, data=data)
        print("Status Code:", response.status_code)
        print("Response JSON:", response.json())
    except Exception as ex:
        print("An error occurred:", ex)

if __name__ == "__main__":
    main()
