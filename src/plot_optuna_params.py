from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_rank
from optuna.visualization import plot_slice
from optuna.visualization import plot_timeline
import optuna
from dash import Dash, dcc, html
import pandas as pd

storage = optuna.storages.RDBStorage(url='sqlite:///optuna_study_dueling.db')
study = optuna.load_study(study_name='RL_study', storage=storage)
print(study.best_params)
print(study.trials[49])
#study.trials_dataframe().to_csv("dataframe.csv")
fig1 = plot_slice(study)
fig2 = plot_param_importances(study)
fig3 = plot_optimization_history(study)
fig4 = plot_timeline(study)
fig5 = plot_parallel_coordinate(study)
# fig6 = plot_intermediate_values(study)
# fig7 = plot_edf(study)
# fig8 = plot_rank(study)


app = Dash()
app.layout = html.Div([
    dcc.Graph(figure=fig1),
    dcc.Graph(figure=fig2),
    dcc.Graph(figure=fig3),
    dcc.Graph(figure=fig4),
    dcc.Graph(figure=fig5),
    # dcc.Graph(figure=fig6),
    # dcc.Graph(figure=fig7),
    # dcc.Graph(figure=fig8),
])

app.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter