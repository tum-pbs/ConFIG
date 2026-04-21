import torch, scipy
from typing import Union, Literal, Callable
from .simulation_paras import *
from scipy.stats.qmc import LatinHypercube
from .simulation_paras import (
    X_START,
    X_END,
    SIMULATION_TIME,
    X_TEST,
    T_TEST,
    INITIAL_CONDITION,
)


class KSSamplerBase:

    def __init__(
        self,
        n_internal: int,
        n_initial: int,
        n_boundary: int,
        device: Union[str, torch.device],
        update_data: bool = False,
        x_start: float = X_START,
        x_end: float = X_END,
        simulation_time: float = SIMULATION_TIME,
        initial_condition: Callable = INITIAL_CONDITION,
        seed: int = 21339,
    ) -> None:
        self.fake_data = [0]
        self.n_initial = n_initial
        self.n_boundary = n_boundary
        self.n_internal = n_internal
        self.device = device
        self.x_start = x_start
        self.x_end = x_end
        self.simulation_time = simulation_time
        self.update_data = update_data
        self.initial_condition = initial_condition
        if not update_data:
            x_i, t_i = self._sample_internal()
            self.sample_internal = lambda: (x_i, t_i)
            x_b_l, t_b_l, x_b_r, t_b_r = self._sample_boundary_v()
            self.sample_boundary = lambda: (x_b_l, t_b_l, x_b_r, t_b_r)
            x_ini, t_ini, h_ini = self._sample_initial()
            self.sample_initial = lambda: (x_ini, t_ini, h_ini)
            self.sample_initial_boundary = lambda: (
                x_ini,
                t_ini,
                h_ini,
                x_b_l,
                t_b_l,
                x_b_r,
                t_b_r,
            )
        else:
            self.sample_internal = self._sample_internal
            self.sample_initial_boundary = self._sample_initial_boundary
            self.sample_boundary = self._sample_boundary
            self.sample_initial = self._sample_initial

    def _sample_initial_boundary(self):
        x_b_l, t_b_l, x_b_r, t_b_r = self._sample_boundary()
        x_new, t_new, h_new = self._sample_initial()
        return x_new, t_new, h_new, x_b_l, t_b_l, x_b_r, t_b_r

    def _sample_initial(self):
        raise NotImplementedError(
            "This method should be implemented by the derived class"
        )

    """   
    def _sample_boundary(self):
        raise NotImplementedError("This method should be implemented by the derived class")    
    """

    def _sample_boundary(self):
        raise NotImplementedError(
            "This method should be implemented by the derived class"
        )

    def _sample_internal(self):
        raise NotImplementedError(
            "This method should be implemented by the derived class"
        )

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.fake_data[idx]


class KSSamplerLH(KSSamplerBase):

    def __init__(
        self,
        n_internal: int,
        n_initial: int,
        n_boundary: int,
        device: Union[str, torch.device],
        update_data: bool = False,
        x_start: float = X_START,
        x_end: float = X_END,
        simulation_time: float = SIMULATION_TIME,
        initial_condition: Callable = INITIAL_CONDITION,
        seed: int = 21339,
    ) -> None:
        self.random_engine_initial = LatinHypercube(d=1, seed=seed)
        self.random_engine_internal = LatinHypercube(d=2, seed=seed)
        self.random_engine_boundary = LatinHypercube(d=1, seed=seed)
        super().__init__(
            n_internal,
            n_initial,
            n_boundary,
            device,
            update_data,
            x_start,
            x_end,
            simulation_time,
            initial_condition,
            seed,
        )

    def _sample_initial(self):
        with torch.no_grad():
            x_initial = (
                torch.tensor(
                    self.random_engine_initial.random(n=self.n_initial)[:, 0],
                    device=self.device,
                    dtype=torch.float32,
                )
                * (self.x_end - self.x_start)
                + self.x_start
            )
            u_initial = self.initial_condition(x_initial)
        return x_initial, torch.zeros_like(x_initial), u_initial

    def _sample_boundary(self):
        with torch.no_grad():
            t_left = (
                torch.tensor(
                    self.random_engine_boundary.random(n=self.n_boundary)[:, 0],
                    device=self.device,
                    dtype=torch.float32,
                )
                * self.simulation_time
            )
            t_right = t_left
            x_left = torch.ones(self.n_boundary, device=self.device) * self.x_start
            x_right = torch.ones(self.n_boundary, device=self.device) * self.x_end
        x_left.requires_grad = True
        x_right.requires_grad = True
        return x_left, t_left, x_right, t_right

    def _sample_internal(self):
        sample = self.random_engine_internal.random(n=self.n_internal)
        x = (
            torch.tensor(sample[:, 0], device=self.device, dtype=torch.float32)
            * (self.x_end - self.x_start)
            + self.x_start
        )
        t = (
            torch.tensor(sample[:, 1], device=self.device, dtype=torch.float32)
            * self.simulation_time
        )
        x.requires_grad = True
        t.requires_grad = True
        return x, t


class KSSamplerMC(KSSamplerBase):

    def __init__(
        self,
        n_internal: int,
        n_initial: int,
        n_boundary: int,
        device: Union[str, torch.device],
        update_data: bool = False,
        x_start: float = X_START,
        x_end: float = X_END,
        simulation_time: float = SIMULATION_TIME,
        initial_condition: Callable = INITIAL_CONDITION,
        seed: int = 21339,
    ) -> None:
        super().__init__(
            n_internal,
            n_initial,
            n_boundary,
            device,
            update_data,
            x_start,
            x_end,
            simulation_time,
            initial_condition,
            seed,
        )

    def _sample_initial(self):
        with torch.no_grad():
            x_initial = (torch.rand(self.n_initial, device=self.device)) * (
                self.x_end - self.x_start
            ) + self.x_start
            u_initial = self.initial_condition(x_initial)
        return x_initial, torch.zeros_like(x_initial), u_initial

    def _sample_boundary(self):
        with torch.no_grad():
            t_left = (
                torch.rand(self.n_boundary, device=self.device) * self.simulation_time
            )
            t_right = t_left
            x_left = torch.ones(self.n_boundary, device=self.device) * self.x_start
            x_right = torch.ones(self.n_boundary, device=self.device) * self.x_end
        x_left.requires_grad = True
        x_right.requires_grad = True
        return x_left, t_left, x_right, t_right

    def _sample_internal(self):
        x = (
            torch.rand(self.n_internal, device=self.device)
            * (self.x_end - self.x_start)
            + self.x_start
        )
        t = (
            torch.rand(self.n_internal, device=self.device, requires_grad=True)
            * self.simulation_time
        )
        x.requires_grad = True
        t.requires_grad = True
        return x, t


def KSSampler(
    n_internal: int,
    n_initial: int,
    n_boundary: int,
    device: Union[str, torch.device],
    update_data: bool = False,
    data_sampler: Literal["latin_hypercube", "monte_carlo"] = "latin_hypercube",
    x_start: float = X_START,
    x_end: float = X_END,
    simulation_time: float = SIMULATION_TIME,
    initial_condition: Callable = INITIAL_CONDITION,
    seed: int = 21339,
):
    if data_sampler == "latin_hypercube":
        return KSSamplerLH(
            n_internal,
            n_initial,
            n_boundary,
            device,
            update_data,
            x_start,
            x_end,
            simulation_time,
            initial_condition,
            seed,
        )
    elif data_sampler == "monte_carlo":
        return KSSamplerMC(
            n_internal,
            n_initial,
            n_boundary,
            device,
            update_data,
            x_start,
            x_end,
            simulation_time,
            initial_condition,
            seed,
        )
    else:
        raise ValueError("Invalid sampler type")


class KSValidationDataSet:

    def __init__(self, dataset_path: str) -> None:
        self.simulation_data = np.load(dataset_path)
        self.xs = X_TEST
        self.ts = T_TEST


class KSValidationDataLoader:

    def __init__(
        self,
        validation_dataset: KSValidationDataSet,
        device: Union[str, torch.device] = "cuda:0",
    ) -> None:
        self.fake_data = [0]
        self.simulation_data = (
            torch.from_numpy(validation_dataset.simulation_data).float().to(device)
        )
        self.xs = (
            torch.from_numpy(validation_dataset.xs)
            .float()
            .to(device)
            .unsqueeze(0)
            .repeat(self.simulation_data.shape[0], 1)
        )
        self.ts = (
            torch.from_numpy(validation_dataset.ts)
            .float()
            .to(device)
            .unsqueeze(1)
            .repeat(1, self.simulation_data.shape[1])
        )

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.fake_data[idx]
