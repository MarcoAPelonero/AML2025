"""One-shot PCA analysis for weight updates across training modes."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

from agent import LinearAgent
from environment import Environment
from trainingUtils import OutOfDistributionTraining, train_episode
from reservoir import build_W_out, initialize_reservoir
from reservoirTrainingUtils import (
    InDistributionTraining as ReservoirInDistributionTraining,
    inference_episode,
    organize_dataset,
)
from stagePredictorReservoir import (
    InDistributionMetaTraining,
    build_meta_weights,
    run_meta_inference_single_theta,
)


def fit_reference_pca(
    ref_weights: Optional[np.ndarray] = None,
    *,
    rounds: int = 1,
    episodes: int = 600,
    time_steps: int = 30,
    n_components: int = 2,
) -> Tuple[PCA, np.ndarray]:
    """Fit PCA on standard out-of-distribution training weights."""
    if ref_weights is None:
        agent = LinearAgent()
        env = Environment()
        _, _, ref_weights = OutOfDistributionTraining(
            agent,
            env,
            rounds=rounds,
            episodes=episodes,
            time_steps=time_steps,
            mode="normal",
            verbose=False,
            return_weights=True,
        )
    ref_weights = np.asarray(ref_weights, dtype=float)
    flat = ref_weights.reshape(ref_weights.shape[0] * ref_weights.shape[1], -1)
    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(flat)
    return pca, ref_weights


def collect_normal_weights(
    lr_list: Sequence[float],
    angles: Sequence[float],
    n_trials: int,
    *,
    time_steps: int = 30,
    show_progress: bool = True,
) -> np.ndarray:
    """Collect final weights after a single REINFORCE episode per configuration."""
    lr_values = list(lr_list)
    angle_values = list(angles)
    template = LinearAgent()
    weight_dim = template.weights.size
    data = np.zeros((len(lr_values), len(angle_values), n_trials, weight_dim), dtype=float)

    total = len(lr_values) * len(angle_values) * max(n_trials, 1)
    progress = tqdm(total=total, desc="Normal (REINFORCE)", leave=False) if show_progress else None

    for lr_idx, lr in enumerate(lr_values):
        for angle_idx, angle in enumerate(angle_values):
            for trial in range(n_trials):
                agent = LinearAgent(
                    learning_rate=lr,
                    temperature=template.temperature,
                )
                env = Environment()
                env.reset(angle)
                agent.reset_parameters()
                train_episode(agent, env, time_steps=time_steps)
                data[lr_idx, angle_idx, trial] = agent.weights.flatten()
                if progress is not None:
                    progress.update(1)
    if progress is not None:
        progress.close()
    return data


def train_reservoir_readout(
    lr: float,
    *,
    time_steps: int = 30,
    rounds: int = 2,
    episodes: int = 600,
    neurons: int = 600,
    show_progress: bool = True,
):
    """Train reservoir readout that maps states to gradients for a given learning rate."""
    agent = LinearAgent(learning_rate=lr)
    env = Environment()
    reservoir = initialize_reservoir(neurons)
    _, _, res_states, grads = ReservoirInDistributionTraining(
        agent,
        env,
        reservoir,
        rounds=rounds,
        episodes=episodes,
        time_steps=time_steps,
        verbose=False,
        bar=show_progress,
    )
    X, Y = organize_dataset(res_states, grads)
    W_out = build_W_out(X, Y)
    trained = copy.deepcopy(reservoir)
    trained.Jout = W_out.T
    trained.reset()
    return trained


def collect_reservoir_weights(
    lr_list: Sequence[float],
    angles: Sequence[float],
    n_trials: int,
    *,
    time_steps: int = 30,
    train_rounds: int = 2,
    train_episodes: int = 600,
    neurons: int = 600,
    show_progress: bool = True,
) -> np.ndarray:
    """Collect weights after one inference episode using reservoir-generated gradients."""
    lr_values = list(lr_list)
    angle_values = list(angles)
    template = LinearAgent()
    weight_dim = template.weights.size
    data = np.zeros((len(lr_values), len(angle_values), n_trials, weight_dim), dtype=float)

    for lr_idx, lr in enumerate(lr_values):
        reservoir = train_reservoir_readout(
            lr,
            time_steps=time_steps,
            rounds=train_rounds,
            episodes=train_episodes,
            neurons=neurons,
            show_progress=show_progress,
        )
        total = len(angle_values) * max(n_trials, 1)
        progress = (
            tqdm(total=total, desc=f"Reservoir lr={lr:g}", leave=False)
            if show_progress
            else None
        )
        for angle_idx, angle in enumerate(angle_values):
            for trial in range(n_trials):
                agent = LinearAgent(
                    learning_rate=lr,
                    temperature=template.temperature,
                )
                env = Environment()
                env.reset(angle)
                agent.reset_parameters()
                inference_episode(agent, env, reservoir, time_steps=time_steps)
                data[lr_idx, angle_idx, trial] = agent.weights.flatten()
                if progress is not None:
                    progress.update(1)
        if progress is not None:
            progress.close()
    return data


def train_stage_predictor(
    lr: float,
    *,
    time_steps: int = 30,
    rounds: int = 3,
    episodes: int = 600,
    neurons: int = 600,
    show_progress: bool = True,
):
    """Train stage-predictor reservoir (meta inference) for a given learning rate."""
    agent = LinearAgent(learning_rate=lr)
    env = Environment()
    reservoir = initialize_reservoir(neurons)
    (
        _,
        total_res_states,
        _,
        total_w_snapshots,
    ) = InDistributionMetaTraining(
        agent,
        env,
        reservoir,
        rounds=rounds,
        episodes=episodes,
        time_steps=time_steps,
        verbose=False,
        bar=show_progress,
    )
    W_meta = build_meta_weights(total_res_states, total_w_snapshots)
    trained = copy.deepcopy(reservoir)
    trained.W_meta = W_meta
    trained.reset()
    return trained


def collect_stage_weights(
    lr_list: Sequence[float],
    angles: Sequence[float],
    n_trials: int,
    *,
    time_steps: int = 30,
    meta_rounds: int = 3,
    meta_episodes: int = 600,
    neurons: int = 600,
    k: int = 1,
    mode: str = "average",
    eta: float = 1.0,
    clip_norm: float = 10.0,
    show_progress: bool = True,
) -> np.ndarray:
    """Collect weights after one stage-predictor meta update."""
    lr_values = list(lr_list)
    angle_values = list(angles)
    template = LinearAgent()
    weight_dim = template.weights.size
    data = np.zeros((len(lr_values), len(angle_values), n_trials, weight_dim), dtype=float)

    for lr_idx, lr in enumerate(lr_values):
        reservoir = train_stage_predictor(
            lr,
            time_steps=time_steps,
            rounds=meta_rounds,
            episodes=meta_episodes,
            neurons=neurons,
            show_progress=show_progress,
        )
        total = len(angle_values) * max(n_trials, 1)
        progress = (
            tqdm(total=total, desc=f"Stage predictor lr={lr:g}", leave=False)
            if show_progress
            else None
        )
        for angle_idx, angle in enumerate(angle_values):
            for trial in range(n_trials):
                agent = LinearAgent(
                    learning_rate=lr,
                    temperature=template.temperature,
                )
                env = Environment()
                env.reset(angle)
                run_meta_inference_single_theta(
                    agent,
                    env,
                    reservoir,
                    theta0=angle,
                    k=k,
                    mode=mode,
                    episodes_total=0,
                    time_steps=time_steps,
                    eta=eta,
                    clip_norm=clip_norm,
                    verbose=False,
                    entropy=True,
                )
                data[lr_idx, angle_idx, trial] = agent.weights.flatten()
                if progress is not None:
                    progress.update(1)
        if progress is not None:
            progress.close()
    return data


def transform_with_pca(weights: np.ndarray, pca: PCA) -> np.ndarray:
    """Project flattened weights into PCA space."""
    if weights.ndim < 2:
        raise ValueError("weights array must have at least 2 dimensions")
    flat = weights.reshape(-1, weights.shape[-1])
    scores = pca.transform(flat)
    return scores.reshape(*weights.shape[:-1], pca.n_components_)


def plot_method_scores(
    method_name: str,
    scores: np.ndarray,
    lr_list: Sequence[float],
    angles: Sequence[float],
    *,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Create scatter plots for PCA scores per learning rate."""
    lr_values = list(lr_list)
    angle_values = list(angles)
    n_lr = len(lr_values)
    if scores.shape[0] != n_lr:
        raise ValueError("scores first dimension must match number of learning rates")
    fig, axes = plt.subplots(1, n_lr, figsize=(4 * n_lr, 4), sharex=True, sharey=True)
    if n_lr == 1:
        axes = [axes]
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % cmap.N) for i in range(len(angle_values))]

    for ax, lr, lr_scores in zip(axes, lr_values, scores):
        for idx_angle, angle in enumerate(angle_values):
            pts = lr_scores[idx_angle]
            if pts.size == 0:
                continue
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                color=colors[idx_angle],
                alpha=0.35,
                s=20,
            )
            mean_xy = pts.mean(axis=0)
            ax.scatter(
                mean_xy[0],
                mean_xy[1],
                color=colors[idx_angle],
                marker="x",
                s=70,
            )
        ax.set_title(f"lr = {lr:g}")
        ax.set_xlabel("PC1")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("PC2")

    handles = [
        plt.Line2D([], [], linestyle="none", marker="o", color=colors[idx], label=f"{angle:.1f} deg")
        for idx, angle in enumerate(angle_values)
    ]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(angle_values), 5))
    fig.suptitle(f"{method_name} one-step weights (PCA)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path is not None:
        fig.savefig(save_path, dpi=240)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def run_experiment(
    lr_list: Sequence[float],
    *,
    n_trials: int = 100,
    angles: Optional[Sequence[float]] = None,
    methods: Sequence[str] = ("normal", "reservoir", "stage"),
    time_steps: int = 30,
    n_components: int = 2,
    pca_rounds: int = 1,
    pca_episodes: int = 600,
    reservoir_rounds: int = 2,
    reservoir_episodes: int = 600,
    stage_rounds: int = 3,
    stage_episodes: int = 600,
    stage_k: int = 1,
    stage_mode: str = "average",
    stage_eta: float = 1.0,
    stage_clip_norm: float = 10.0,
    reservoir_neurons: int = 600,
    output_dir: Optional[Path | str] = None,
    show_progress: bool = True,
    plot: bool = True,
    show_plots: bool = False,
) -> Dict[str, Any]:
    """Run the full experiment and optionally persist results."""
    allowed_methods = {"normal", "reservoir", "stage"}
    selected = tuple(m.lower() for m in methods)
    if not set(selected).issubset(allowed_methods):
        raise ValueError(f"methods must be subset of {allowed_methods}")

    lr_values = list(lr_list)
    angle_values = (
        [0.0, 22.5, 45.0, 67.5, 90.0] if angles is None else list(angles)
    )

    pca, ref_weights = fit_reference_pca(
        rounds=pca_rounds,
        episodes=pca_episodes,
        time_steps=time_steps,
        n_components=n_components,
    )

    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path / "pca_reference.npz",
            mean=pca.mean_,
            components=pca.components_,
            explained_variance=pca.explained_variance_,
            singular_values=pca.singular_values_,
        )

    results: Dict[str, Any] = {
        "lr_list": np.array(lr_values, dtype=float),
        "angles": np.array(angle_values, dtype=float),
        "pca": pca,
        "reference_weights": ref_weights,
        "methods": {},
    }

    if "normal" in selected:
        norm_weights = collect_normal_weights(
            lr_values,
            angle_values,
            n_trials,
            time_steps=time_steps,
            show_progress=show_progress,
        )
        norm_scores = transform_with_pca(norm_weights, pca)
        results["methods"]["normal"] = {
            "weights": norm_weights,
            "scores": norm_scores,
        }
        if output_path is not None:
            np.savez_compressed(
                output_path / "normal_weights.npz",
                weights=norm_weights,
                scores=norm_scores,
                lr=np.array(lr_values, dtype=float),
                angles=np.array(angle_values, dtype=float),
            )
        if plot:
            plot_method_scores(
                "Normal",
                norm_scores,
                lr_values,
                angle_values,
                save_path=None if output_path is None else output_path / "normal_scatter.png",
                show=show_plots,
            )

    if "reservoir" in selected:
        res_weights = collect_reservoir_weights(
            lr_values,
            angle_values,
            n_trials,
            time_steps=time_steps,
            train_rounds=reservoir_rounds,
            train_episodes=reservoir_episodes,
            neurons=reservoir_neurons,
            show_progress=show_progress,
        )
        res_scores = transform_with_pca(res_weights, pca)
        results["methods"]["reservoir"] = {
            "weights": res_weights,
            "scores": res_scores,
        }
        if output_path is not None:
            np.savez_compressed(
                output_path / "reservoir_weights.npz",
                weights=res_weights,
                scores=res_scores,
                lr=np.array(lr_values, dtype=float),
                angles=np.array(angle_values, dtype=float),
            )
        if plot:
            plot_method_scores(
                "Reservoir",
                res_scores,
                lr_values,
                angle_values,
                save_path=None if output_path is None else output_path / "reservoir_scatter.png",
                show=show_plots,
            )

    if "stage" in selected:
        stage_weights = collect_stage_weights(
            lr_values,
            angle_values,
            n_trials,
            time_steps=time_steps,
            meta_rounds=stage_rounds,
            meta_episodes=stage_episodes,
            neurons=reservoir_neurons,
            k=stage_k,
            mode=stage_mode,
            eta=stage_eta,
            clip_norm=stage_clip_norm,
            show_progress=show_progress,
        )
        stage_scores = transform_with_pca(stage_weights, pca)
        results["methods"]["stage"] = {
            "weights": stage_weights,
            "scores": stage_scores,
        }
        if output_path is not None:
            np.savez_compressed(
                output_path / "stage_weights.npz",
                weights=stage_weights,
                scores=stage_scores,
                lr=np.array(lr_values, dtype=float),
                angles=np.array(angle_values, dtype=float),
            )
        if plot:
            plot_method_scores(
                "Stage predictor",
                stage_scores,
                lr_values,
                angle_values,
                save_path=None if output_path is None else output_path / "stage_scatter.png",
                show=show_plots,
            )

    return results


def plot_all_methods_from_saved(
    output_dir: Path | str,
    *,
    show: bool = False,
    filename: str = "all_methods_scatter.png",
) -> plt.Figure:
    """Load saved results and produce a combined figure with all methods x learning rates.

    Rows = methods (Normal, Reservoir, Stage predictor)
    Columns = learning rates
    """
    output_path = Path(output_dir)
    files = {
        "normal": output_path / "normal_weights.npz",
        "reservoir": output_path / "reservoir_weights.npz",
        "stage": output_path / "stage_weights.npz",
    }

    loaded = {}
    for key, f in files.items():
        if f.exists():
            data = np.load(f, allow_pickle=False)
            loaded[key] = {
                "scores": data["scores"],
                "lr": data["lr"],
                "angles": data["angles"],
            }

    if not loaded:
        raise FileNotFoundError(f"No saved results found in {output_path}")

    # Use the first available method to derive lr and angles order
    first_key = next(iter(loaded))
    lr_values = loaded[first_key]["lr"]
    angle_values = loaded[first_key]["angles"]
    n_lr = len(lr_values)

    method_order = [m for m in ("normal", "reservoir", "stage") if m in loaded]
    method_titles = {
        "normal": "Normal",
        "reservoir": "Reservoir",
        "stage": "Stage predictor",
    }
    n_rows = len(method_order)

    fig, axes = plt.subplots(
        n_rows,
        n_lr,
        figsize=(4 * n_lr, 4 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % cmap.N) for i in range(len(angle_values))]

    for r, method in enumerate(method_order):
        scores = loaded[method]["scores"]  # shape: (n_lr, n_angles, n_trials, 2)
        for c, lr in enumerate(lr_values):
            ax = axes[r, c]
            # Plot points per angle
            for a_idx, angle in enumerate(angle_values):
                pts = scores[c, a_idx]
                if pts.size == 0:
                    continue
                ax.scatter(pts[:, 0], pts[:, 1], color=colors[a_idx], alpha=0.35, s=20)
                mean_xy = pts.mean(axis=0)
                ax.scatter(mean_xy[0], mean_xy[1], color=colors[a_idx], marker="x", s=70)

            if r == 0:
                ax.set_title(f"lr = {lr:g}")
            if c == 0:
                ax.set_ylabel("PC2")
                ax.annotate(
                    method_titles[method],
                    xy=(0, 1.02),
                    xycoords="axes fraction",
                    ha="left",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                )
            ax.set_xlabel("PC1")
            ax.grid(alpha=0.2)

    handles = [
        plt.Line2D([], [], linestyle="none", marker="o", color=colors[idx], label=f"{angle:.1f} deg")
        for idx, angle in enumerate(angle_values)
    ]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(angle_values), 5))
    fig.suptitle("All methods one-step weights (PCA)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    save_path = output_path / filename
    fig.savefig(save_path, dpi=240)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def run_post_experiment(
    output_dir: Path | str,
    *,
    show: bool = False,
    filename: str = "all_methods_scatter_independent_axes.png",
) -> plt.Figure:
    """Rebuild the combined figure from saved data with independent axes per subplot.

    Loads the saved NPZ files (normal/reservoir/stage) from output_dir and creates a
    methods x learning-rate grid, without sharing x/y limits between subplots.
    """
    output_path = Path(output_dir)
    files = {
        "normal": output_path / "normal_weights.npz",
        "reservoir": output_path / "reservoir_weights.npz",
        "stage": output_path / "stage_weights.npz",
    }

    loaded = {}
    for key, f in files.items():
        if f.exists():
            data = np.load(f, allow_pickle=False)
            loaded[key] = {
                "scores": data["scores"],
                "lr": data["lr"],
                "angles": data["angles"],
            }

    if not loaded:
        raise FileNotFoundError(f"No saved results found in {output_path}")

    # Use the first available method to derive lr and angles order
    first_key = next(iter(loaded))
    lr_values = loaded[first_key]["lr"]
    angle_values = loaded[first_key]["angles"]
    n_lr = len(lr_values)

    method_order = [m for m in ("normal", "reservoir", "stage") if m in loaded]
    method_titles = {
        "normal": "Normal",
        "reservoir": "Reservoir",
        "stage": "Stage predictor",
    }
    n_rows = len(method_order)

    # Independent axes: do not share x or y
    fig, axes = plt.subplots(
        n_rows,
        n_lr,
        figsize=(4 * n_lr, 4 * n_rows),
        sharex=False,
        sharey=False,
        squeeze=False,
    )

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % cmap.N) for i in range(len(angle_values))]

    for r, method in enumerate(method_order):
        scores = loaded[method]["scores"]  # shape: (n_lr, n_angles, n_trials, 2)
        for c, lr in enumerate(lr_values):
            ax = axes[r, c]
            for a_idx, angle in enumerate(angle_values):
                pts = scores[c, a_idx]
                if pts.size == 0:
                    continue
                ax.scatter(pts[:, 0], pts[:, 1], color=colors[a_idx], alpha=0.35, s=20)
                mean_xy = pts.mean(axis=0)
                ax.scatter(mean_xy[0], mean_xy[1], color=colors[a_idx], marker="x", s=70)

            # Independent autoscale per subplot
            ax.margins(0.1)  # small padding around data
            ax.grid(alpha=0.2)
            ax.set_xlabel("PC1")
            if r == 0:
                ax.set_title(f"lr = {lr:g}")
            if c == 0:
                ax.set_ylabel("PC2")
                ax.annotate(
                    method_titles[method],
                    xy=(0, 1.02),
                    xycoords="axes fraction",
                    ha="left",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                )

    handles = [
        plt.Line2D([], [], linestyle="none", marker="o", color=colors[idx], label=f"{angle:.1f} deg")
        for idx, angle in enumerate(angle_values)
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=min(len(angle_values), 5))
    fig.suptitle("All methods one-step weights (PCA)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    save_path = output_path / filename
    fig.savefig(save_path, dpi=240)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def main() -> None:
    """Example execution with increased samples and 4 angles."""
    lr_values = [0.02, 0.2, 2]
    out_dir = Path("one_shot_param_dynamics")
    run_experiment(
        lr_values,
        n_trials=800,  # more scatter points
        angles=[0.0, 22.5, 45.0, 67.5, 90.0],  # 5 angles
        plot=True,
        show_plots=False,
        output_dir=out_dir,
    )
    # Build one figure with all methods x learning rates
    plot_all_methods_from_saved(out_dir, show=False, filename="all_methods_scatter.png")
    print("Experiment finished. Results saved in 'one_shot_param_dynamics'.")
    print("Combined plot saved to 'one_shot_param_dynamics/all_methods_scatter.png'.")

def post_process() -> None:
    """Run post-experiment plotting with independent axes."""
    out_dir = Path("one_shot_param_dynamics")
    run_post_experiment(
        out_dir,
        show=True,
        filename="all_methods_scatter_independent_axes.png",
    )
    print("Post-processing finished. Independent axes plot saved to 'one_shot_param_dynamics/all_methods_scatter_independent_axes.png'.")

if __name__ == "__main__":
    # main()
    post_process()
