from abc import ABC, abstractmethod
import xarray as xr


class Metric(ABC):
    """Base class for all metrics."""

    @abstractmethod
    def compute(self, forecast: xr.Dataset, ground_truth: xr.Dataset) -> xr.Dataset:
        """Compute metric between forecast and ground truth."""


class RMSE(Metric):
    def compute(self, forecast: xr.Dataset, ground_truth: xr.Dataset) -> xr.Dataset:
        """Compute Root Mean Square Error between forecast and ground truth.
        According to equation 2 in https://arxiv.org/pdf/2308.15560.

        For each station (s) and lead time (l):
        RMSE(s,l) = sqrt(1/T * sum_t[(f_{s,t,l} - o_{s,t})^2])

        where:
        - t: time index
        - T: total number of time steps
        - f: forecast
        - o: observation (ground truth)
        """
        rmse = {}
        for var in forecast.data_vars:
            rmse[var] = ((forecast[var] - ground_truth[var]) ** 2).mean(
                "init_time", skipna=True
            ) ** 0.5

        return xr.Dataset(rmse).expand_dims(metric=["rmse"])


class MBE(Metric):
    def compute(self, forecast: xr.Dataset, ground_truth: xr.Dataset) -> xr.Dataset:
        """Compute Mean Bias Error between forecast and ground truth.
        According to equation 6 in https://arxiv.org/pdf/2308.15560.

        For each station (s) and lead time (l):
        MBE(s,l) = 1/T * sum_t[f_{s,t,l} - o_{s,t}]

        where:
        - t: time index
        - T: total number of time steps
        - f: forecast
        - o: observation (ground truth)
        """
        mbe = {}
        for var in forecast.data_vars:
            mbe[var] = (forecast[var] - ground_truth[var]).mean(
                "init_time", skipna=True
            )

        return xr.Dataset(mbe).expand_dims(metric=["mbe"])


AVAILABLE_METRICS = {
    "rmse": RMSE(),
    "mbe": MBE(),
}
