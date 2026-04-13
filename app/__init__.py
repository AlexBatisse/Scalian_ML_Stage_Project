from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'dev-key-change-in-prod'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///energy.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)

    from app.routes.dashboard import dashboard_bp
    from app.routes.analyse import analyse_bp
    from app.routes.alertes import alertes_bp

    app.register_blueprint(dashboard_bp)
    app.register_blueprint(analyse_bp)
    app.register_blueprint(alertes_bp)

    return app