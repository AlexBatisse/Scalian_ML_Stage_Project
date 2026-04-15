from flask import Blueprint, render_template

alertes_bp = Blueprint('alertes', __name__)

@alertes_bp.route('/alertes')
def alertes():
    return render_template('alertes.html', titre="Alertes")
