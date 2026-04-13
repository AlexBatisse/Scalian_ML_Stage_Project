from flask import Blueprint, render_template, request

analyse_bp = Blueprint('analyse', __name__)

@analyse_bp.route('/analyse')
def analyse():
    date_debut = request.args.get('date_debut', '')
    date_fin = request.args.get('date_fin', '')
    return render_template('analyse.html', titre="Analyse", 
                           date_debut=date_debut, date_fin=date_fin)