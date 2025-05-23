"""
Backend server for processing images.
This server receives:
  - a buildings_image (which should include one green pixel for takeoff and one red pixel for landing),
  - a satellite_image,
  - (the diagonal coordinates are fixed as below)

The server processes the buildings image by applying skeletonization, junction detection,
graph construction and pathfinding (using Dijkstra). It then converts the computed path's
pixel coordinates into real-world coordinates (using the fixed diagonal), and overlays the
path on both the buildings and satellite images.
"""

import os
from flask import Flask, request
from flask_cors import CORS
from services import mission_service

app = Flask(__name__)
CORS(app, origins=["https://www.skyops.co.il"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'static', 'outputs')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/api/create-mission", methods=["POST"])
def create_mission_route():
    return mission_service.create_mission(request)

# Color constants
GREEN = (0, 255, 0)
RED = (0, 0, 255)

# Section: Run the Flask application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
