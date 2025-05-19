from flask import request
from services import mission_service

def register_routes(app):
    @app.route("/api/create-mission", methods=["POST"])
    def create_mission_route():
        return mission_service.create_mission(request)
