from sklearn.model_selection import RandomizedSearchCV, ParameterSampler
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

class CustomRandomizedSearchCV(RandomizedSearchCV):
    def fit(self, X, y=None, groups=None, *args, **kwargs):
        self._progress_bar = tqdm(total=self.n_iter)
        super().fit(X, y, groups, *args, **kwargs)
        self._progress_bar.close()

    def _run_search(self, evaluate_candidates):
        evaluate_candidates(ParameterSampler(
            self.param_distributions, self.n_iter,
            random_state=self.random_state))

        self._progress_bar.update(1)
