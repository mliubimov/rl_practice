from run import run
import optuna
import sys
import datetime
import signal
import copy
from typing import Dict, Any, Callable, Tuple, Union
import statistics


class OptunaOptimizer:
    def __init__(
            self, params: Dict[str, Any]) -> None:
        self.params: Dict[str, Any] = params
        self.study_name = params.get("study_name", 'RL_study')
        self.storage_path = params.get("storage_path", "sqlite:///optuna_study_double.db")
        self.study: Union[optuna.Study, None] = None

    def optimize_RL(self) -> Dict[str, Any]:
        def objective(trial: optuna.Trial) -> float:
            for param_name, (param_type, param_values) in self.params['optuna_params'].items():
                if callable(param_values):
                    param_values = param_values(self.params)
                if param_type == "continuous":
                    self.params[param_name] = trial.suggest_float(param_name, *param_values)
                elif param_type == "discrete":
                    self.params[param_name] = trial.suggest_categorical(param_name, param_values)

            self.params['load_weights'] = False
            self.params['train'] = True
            # Run the RL task
            scores = run(self.params, False, False, True)

            if self.params['params_search_criteria'] == "test_mean":

                params: Dict[str, Any] = copy.deepcopy(self.params)
                params['episodes'] = self.params['val_episodes']
                params['epsilon_decay_linear'] = 1
                params['load_weights'] = True

                scores = run(params, False, False, False)
                mean, stdev = statistics.mean(scores), statistics.stdev(scores)

                print(f'Total score: {sum(scores)}   Mean: {mean}   Std dev: {stdev}')
                return mean
            elif self.params['params_search_criteria'] == "top_ten_scores":
                top_ten_elements = sorted(scores, reverse=True)[:10]
                top_ten_sum = sum(top_ten_elements)
                return top_ten_sum
            else:
                raise "Unknown criteria " + self.params['params_search_criteria']

        def signal_handler(sig: int, frame: Any) -> None:
            print("\nInterrupt received, saving study...")
            self.study.stop()
            self.get_final_params()
            sys.exit(0)

        # Set up signal handling for interruption
        signal.signal(signal.SIGINT, signal_handler)

        # Use RDBStorage to store study progress in an SQLite database
        storage: optuna.storages.RDBStorage = optuna.storages.RDBStorage(url=self.storage_path)

        # Try to load an existing study or create a new one
        try:
            self.study = optuna.load_study(study_name=self.study_name, storage=storage)
            print(f"Resuming study '{self.study_name}'")
        except KeyError:
            # If study doesn't exist, create a new one
            self.study = optuna.create_study(study_name=self.study_name, storage=storage, direction='maximize')
            print(f"Creating new study '{self.study_name}'")

        try:
            # Optimize the study, and it will save progress in the database
            self.study.optimize(objective, n_trials=self.params['n_trials'])
        except KeyboardInterrupt:
            print("\nOptimization interrupted by user.")
            self.study.stop()

        self.get_final_params()

    def get_final_params(self) -> None:
        if self.study is not None:
            print("\nParameter Importance:")
            print(optuna.importance.get_param_importances(self.study))
            print('Current best parameters:')
            for param_name in self.params['optuna_params']:
                print(f"{param_name}: {self.study.best_params[param_name]}")
