#!/usr/bin/env python
"""
Risk–Controlling Prediction Sets for Regression via Conformal Inference

This simulation uses a regression problem (generated via sklearn’s make_regression)
and a linear regression model. A base conformal interval is computed via the (1 – nominal_alpha)
quantile of the calibration absolute residuals. We then “risk–control” the interval by adding an extra
buffer t, so that the prediction interval becomes [f(x) – (q0+t), f(x) + (q0+t)].

For a given candidate t (searched on a grid from 0 to 30 with 101 steps), the empirical miscoverage risk
on the calibration set is computed and a concentration bound is added.
A fixed probability threshold delta (δ) and a target risk level are used. The smallest t such that
(empirical risk + bound) ≤ target_risk is selected.

This code is modular in that the concentration bound is computed by the function `get_bound`,
which easily allows new bounds to be added. Finally, the script produces calibration plots (with
all bound types overlaid) and boxplots comparing the distributions of chosen t, test risk, coverage,
and average prediction interval size.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import ot
import re

import torch as pt
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm

from skwdro.torch import robustify
from skwdro.solvers.oracle_torch import DualLoss


# For reproducibility
# np.random.seed(42)
# pt.manual_seed(42)

rho_list = [0.5, 1.0, 2.0, 4.0 ]

folder = "./RCPS_shift_Out/"
plt.style.use("paper.mplstyle")


# ----------------------
# Concentration Bound Functions
# ----------------------
def Hoeffding_bound(n, delta):
    """Hoeffding upper bound for 0-1 losses."""
    return np.sqrt(np.log(1/delta) / (2 * n))

def Bernstein_bound(errors, n, delta):
    """Empirical Bernstein bound for 0-1 losses."""
    r_hat = np.mean(errors)
    sigma2 = np.var(errors)
    return np.sqrt(2 * sigma2 * np.log(2/delta) / n) + (7 * np.log(2/delta)) / (3 * max(n - 1, 1))

def WDRO_special_bound(residuals_cal, lam_val, n, rho=1.0):
    "WDRO bound in the special case of Section 3.4.2"
    residuals_sorted = np.sort(residuals_cal)[::-1]  # descending order
    i_0 = -1
    for i, r in enumerate(residuals_sorted):
        if r >= lam_val:
            i_0 = i
        else:
            break
    trial_gammas = 1/(lam_val - residuals_sorted[i_0+1:])**2
    loss = np.zeros(np.size(trial_gammas))
    for i, g in enumerate(trial_gammas):
        j = i_0+1
        s = 0
        while (j < n) and (residuals_sorted[j] >= lam_val - 1/np.sqrt(g)):
            term = 1 - g*(lam_val - residuals_sorted[j])**2
            s += term
            j += 1
        loss[i] = g*rho**2 + (i_0+1)/n + s/n 
    bound_val = min(1, np.min(loss))
    return bound_val



# ----------------------
# Main WDRO Function
# ----------------------
def WDRO_entropy_bound(f, X_cal, y_cal, lam, n, rho=1.0):
    """
    WDRO bound w/ skwdro without batching
    """

    # Convert numpy arrays to tensors
    X_cal = pt.from_numpy(X_cal).to(pt.float32)
    y_cal = pt.from_numpy(y_cal).to(pt.float32)
    y_cal = y_cal.unsqueeze(-1)

    # Dataset
    dataset = DataLoader(TensorDataset(X_cal, y_cal), batch_size=n, shuffle=True)

    device = "cuda" if pt.cuda.is_available() else "cpu"

    # Extract weights and bias from the trained sklearn model
    sk_weights = f.coef_.flatten()  # Shape: (d,)
    sk_bias = f.intercept_           # Shape: (1,)

    # Define a PyTorch model with fixed weights
    class FixedLinearModel(nn.Module):
        def __init__(self, weights, bias):
            super(FixedLinearModel, self).__init__()
            self.linear = nn.Linear(in_features=weights.shape[0], out_features=1)
            
            # Set the weights and bias
            with pt.no_grad():
                self.linear.weight.copy_(pt.tensor(weights).unsqueeze(0))  # Shape: (1, d)
                self.linear.bias.copy_(pt.tensor(bias))                    # Shape: (1,)

            # Disable gradient computation for weights and bias
            self.linear.weight.requires_grad = False
            self.linear.bias.requires_grad = False

        def forward(self, x):
            self.linear.weight.requires_grad = False
            self.linear.bias.requires_grad = False
            pred = self.linear(x)
            return pred



    # Example conformal loss function 
    def conformal_loss(output: pt.Tensor, target: pt.Tensor) -> pt.Tensor:

        # ## Option 1
        # loss = F.l1_loss(output,target,reduction='none')>lam
        # loss = loss.float()

        # ## Option 2
        # loss =  F.relu(F.l1_loss(output,target,reduction='none') - lam)

        ## Option 3
        loss = F.sigmoid( 30.0*(F.l1_loss(output,target,reduction='none') - lam) + 3 ) # Boosted sigmoid

        return loss.squeeze(-1)



    model_conf = FixedLinearModel(sk_weights, sk_bias).to(device)  # Our regression model


    WDRO_conf = robustify(
        conformal_loss,
        model_conf,
        pt.tensor(rho),
        X_cal,
        y_cal
    )


    epochs = 10

    optimizer = pt.optim.AdamW(params=WDRO_conf.parameters(),lr=0.1)

    # optimizer = pt.optim.SGD(params=WDRO_conf.parameters())

    # Training loop
    iterator = tqdm(range(epochs), position=0, desc='Epochs', leave=False)
    losses = []
    for epoch in iterator:

        for batch_x, batch_y in tqdm(dataset, position=1, desc='Sample', leave=False):

            optimizer.zero_grad()
            loss = WDRO_conf(batch_x, batch_y, reset_sampler=True)
            loss.backward()
            optimizer.step()
            if WDRO_conf.lam < 0.:
                with pt.no_grad():
                    WDRO_conf._lam *= 0.



    wdro_loss = WDRO_conf(X_cal,y_cal, reset_sampler=True).item()
    bound_value = max(0, min(wdro_loss, 1))
    return bound_value


# ----------------------
# Modular Bound Function
# ----------------------
def get_bound(errors, n, delta, bound_type):
    """
    Modular function to compute the concentration bound.
    New bounds can be added by extending this function.
    """
    bound_functions = {
        "Hoeffding": lambda: Hoeffding_bound(n, delta),
        "Bernstein": lambda: Bernstein_bound(errors, n, delta),
        # Additional bounds can be added here.
    }
    if bound_type not in bound_functions:
        raise ValueError(f"Unknown bound_type: {bound_type}")
    return bound_functions[bound_type]()


# ----------------------
# Helper Functions for Regression Prediction Intervals
# ----------------------
def compute_baseline_quantile(residuals, nominal_alpha):
    """Compute the (1 - nominal_alpha) quantile of calibration residuals."""
    return np.quantile(residuals, 1 - nominal_alpha)

def prediction_interval(fx, q0, t):
    """Return the prediction interval [fx - (q0+t), fx + (q0+t)]."""
    return fx - (q0 + t), fx + (q0 + t)

def evaluate_regression(f, X, y, q0, t):
    """
    Evaluate prediction intervals [f(x)-(q0+t), f(x)+(q0+t)] on data (X,y).
    Returns:
      - miscoverage risk (fraction of points not covered)
      - average interval size (width)
      - coverage (1 - risk)
    """
    predictions = f.predict(X)
    lower = predictions - (q0 + t)
    upper = predictions + (q0 + t)
    errors = ((y < lower) | (y > upper)).astype(int)
    risk = np.mean(errors)
    coverage = 1 - risk
    avg_size = np.mean(upper - lower)
    return risk, avg_size, coverage

def choose_threshold_regression(f, X_cal, y_cal, q0, target_risk, delta, bound_type, grid_t):
    """
    For a trained regression model f and calibration data, scan over a grid of extra buffer t values.
    For each candidate t, compute the miscoverage risk on the calibration set when using the interval
      [f(x)-(q0+t), f(x)+(q0+t)].
    Then add a concentration bound (via get_bound) with probability delta.
    Return the smallest t such that (empirical risk + bound) <= target_risk,
    along with the arrays of grid_t, empirical risks, bounds, and upper bounds.
    """
    n = len(X_cal)
    predictions_cal = f.predict(X_cal)
    residuals_cal = np.abs(y_cal - predictions_cal)
    
    cal_risks = []
    bounds = []
    upper_bounds = []
    
    for t in grid_t:
        errors = (residuals_cal > (q0 + t)).astype(int)
        r_hat = np.mean(errors)
        if bound_type in ["Hoeffding", "Bernstein"]:
            bnd = get_bound(errors, n, delta, bound_type)
            cal_risks.append(r_hat)
            bounds.append(bnd)
            upper_bounds.append(r_hat + bnd)
        elif bound_type == "WDRO":
            bnd = WDRO_special_bound(residuals_cal, q0 + t, n, rho=1.5)
            cal_risks.append(r_hat)
            bounds.append(bnd - r_hat)
            upper_bounds.append(bnd)
        elif bound_type == "SKWDRO":
            bnd = WDRO_entropy_bound(f, X_cal, y_cal, q0 + t, n, rho=.15)
            cal_risks.append(r_hat)
            bounds.append(bnd - r_hat)
            upper_bounds.append(bnd)
        elif re.match(r"rho=([0-9]*\.?[0-9]+)", bound_type):
            rho = float(re.match(r"rho=([0-9]*\.?[0-9]+)", bound_type).group(1))
            bnd = WDRO_special_bound(residuals_cal, q0 + t, n, rho=rho)
            cal_risks.append(r_hat)
            bounds.append(bnd - r_hat)
            upper_bounds.append(bnd)
        else:
            raise ValueError(f"Unknown bound_type: {bound_type}")
        
    cal_risks = np.array(cal_risks)
    bounds = np.array(bounds)
    upper_bounds = np.array(upper_bounds)
    feasible = grid_t[upper_bounds <= target_risk]
    chosen_t = np.min(feasible) if feasible.size > 0 else np.max(grid_t)
    return chosen_t, grid_t, cal_risks, bounds, upper_bounds






# ----------------------
# Simulation and Plotting Functions
# ----------------------
def simulation_run_regression(n_samples=1000, train_frac=0.5, cal_frac=0.25, test_frac=0.25,
                              nominal_alpha=0.1, target_risk=0.1, delta=0.1,
                              bound_types=["Hoeffding", "Bernstein", "WDRO"], grid_t=None):
    """
    Run one simulation for a regression problem:
      - Generate data and split into training, calibration, and test sets.
      - Train a regression model.
      - Compute the baseline quantile q0 from calibration residuals.
      - For each bound type, choose the extra buffer t using choose_threshold_regression.
      - Evaluate the resulting prediction intervals on the test set.
    Returns a dictionary (per bound type) with the chosen t and evaluation metrics.
    """
    X, y = make_regression(n_samples=n_samples, n_features=5, n_informative=3,
                           noise=20, random_state=42)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_frac, random_state=None)
    X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp,
                                                    test_size=test_frac/(cal_frac+test_frac), random_state=None)
    
    X_shift = X_test.copy()
    y_shift = y_test.copy()
    per_coord_shift = np.zeros(X_shift.shape)
    per_coord_shift[:,0] = 0.25
    X_shift = X_shift  + per_coord_shift + 0.25*np.random.randn(X_shift.shape[0],X_shift.shape[1])
    y_shift = y_shift # + 1 + 1.0*np.random.randn(y_shift.size)

    Xy_test     = np.hstack((X_test,y_test.reshape(-1,1)))
    Xy_shift    = np.hstack((X_shift,y_shift.reshape(-1,1)))
    w2_shift    = np.sqrt(ot.emd2([],[],ot.dist(Xy_test,Xy_shift, "sqeuclidean")))

    print(f"Distribution shift in W2 distance = {w2_shift}")

    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_cal_pred = model.predict(X_cal)
    residuals_cal = np.abs(y_cal - y_cal_pred)
    q0 = compute_baseline_quantile(residuals_cal, nominal_alpha)
    
    results = {}
    for btype in bound_types:
        t_chosen, grid_t_arr, cal_risks, bnds, up_bounds = choose_threshold_regression(
            model, X_cal, y_cal, q0, target_risk, delta, bound_type=btype, grid_t=grid_t
        )
        test_risk, avg_size, coverage = evaluate_regression(model, X_test, y_test, q0, t_chosen)
        shift_risk, _,  shift_coverage = evaluate_regression(model, X_shift, y_shift, q0, t_chosen)
        print(f"{shift_coverage=}")
        results[btype] = {
            "t_chosen": t_chosen,
            "cal_risks": cal_risks,
            "bounds": bnds,
            "upper_bounds": up_bounds,  
            "grid_t": grid_t_arr,
            "test_risk": test_risk,
            "shift_risk": shift_risk,
            "coverage": coverage,
            "shift_coverage": shift_coverage,
            "avg_interval_size": avg_size,
            "q0": q0
        }
    return results, model, w2_shift

def plot_calibration_results_all(results_dict, target_risk):
    """
    Plot calibration curves for all bound types on a single figure.
    For each bound type, plot the empirical calibration risk and (risk+bnd) vs. t.
    """
    plt.figure(figsize=(10, 6))
    for btype, res in results_dict.items():
        plt.plot(res["q0"]+res["grid_t"], res["upper_bounds"], marker='s', linestyle='--', label=f"{btype} Bound")
        # plt.axvline(res["t_chosen"], ls='--', label=f"{btype}: Chosen t = {res['t_chosen']:.2f}")
    plt.plot(res["q0"]+res["grid_t"], res["cal_risks"], marker='', linestyle='-', label=f"Empirical Conformal Risk")
    plt.axhline(target_risk, color='red', ls='--', label=f"Target risk")
    plt.xlabel("Prediction size $\lambda$")
    plt.ylabel(" Risk / Upper Bound")
    # plt.title("Calibration Curves for Different Bounds")
    plt.legend() # bbox_to_anchor=(1.05, 1), loc="upper left"
    plt.tight_layout()
    plt.savefig(folder+"Bounds_shifted.pdf")
    #plt.show()

def plot_distribution_comparisons(rep_results, bound_types, delta, target_risk):
    """
    Create boxplots comparing the distributions of chosen t, test risk, coverage, and interval size
    across different bound types.
    """
    metrics = [ "coverage", "shift_coverage", "avg_interval_size"]
    titles = {
       # "t_chosen": "Chosen Extra Buffer t",
        "coverage": "Test Coverage",
        "shift_coverage": "Coverage after distribution shift",
        "avg_interval_size": "Prediction Interval Size"
    }

    bound_types_names = ["Hoeffding", "Bernstein"]
    for rho in rho_list:
        bound_types_names.append(f"$\\rho={rho}$")
    
    fig, axs = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5))
    
    for i, metric in enumerate(metrics):
        data = [rep_results[btype][metric] for btype in bound_types]
        axs[i].boxplot(data, tick_labels=bound_types_names,whis=(delta,1-delta))
        axs[i].set_title(titles[metric])
        #axs[i].set_ylabel("Coverage" if metric=="coverage" else metric)
    # plt.suptitle("Metrics Across Bound Types")
    axs[0].axhline(1-target_risk, color='red', ls='--')
    axs[0].set_ylim(1-target_risk-0.04,1.0)
    axs[1].axhline(1-target_risk, color='red', ls='--')
    axs[1].set_ylim(1-target_risk-0.04,1.0)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(folder+"Results_shifted.pdf")
    #plt.show()



def main():
    n_replications = 100
    n_samples = 1000
    target_risk = 0.1              # Desired controlled miscoverage risk
    nominal_alpha = target_risk     # Base miscoverage level (95% nominal coverage)
    delta = 0.05                     # Confidence parameter for the concentration bound
    bound_types = ["Hoeffding", "Bernstein"]
    for rho in rho_list:
        bound_types.append(f"rho={rho}")
    grid_t = np.linspace(0, 50, 51)  # Extended grid for t
    
    results_example, model, _ = simulation_run_regression(n_samples=n_samples, nominal_alpha=nominal_alpha,
                                                        target_risk=target_risk, delta=delta,
                                                        bound_types=bound_types, grid_t=grid_t)
    plot_calibration_results_all(results_example, target_risk)
    
    rep_results = {btype: {"t_chosen": [], "coverage": [], "shift_coverage": [], "avg_interval_size": []}
                   for btype in bound_types}
    
    shifts = []

    for rep in range(n_replications):
        if int(100*rep/n_replications)%10 == 0:
            print(f"{rep+1}/{n_replications} runs")
        sim_res, _, w2_shift = simulation_run_regression(n_samples=n_samples, nominal_alpha=nominal_alpha,
                                                target_risk=target_risk, delta=delta,
                                                bound_types=bound_types, grid_t=grid_t)
        shifts.append(w2_shift)

        for btype in bound_types:
            rep_results[btype]["t_chosen"].append(sim_res[btype]["t_chosen"])
            rep_results[btype]["coverage"].append(sim_res[btype]["coverage"])
            rep_results[btype]["shift_coverage"].append(sim_res[btype]["shift_coverage"])
            rep_results[btype]["avg_interval_size"].append(sim_res[btype]["avg_interval_size"])
    
    for btype in bound_types:
        avg_t = np.mean(rep_results[btype]["t_chosen"])
        avg_coverage = np.mean(rep_results[btype]["coverage"])
        quant_coverage = np.quantile(rep_results[btype]["coverage"],delta)
        avg_shift_coverage = np.mean(rep_results[btype]["shift_coverage"])
        quant_shift_coverage = np.quantile(rep_results[btype]["shift_coverage"],delta)
        avg_size = np.mean(rep_results[btype]["avg_interval_size"])
        print(f"Over {n_replications} replications, bound type '{btype}':")
        # print(f"  Average chosen t: {avg_t:.3f}")
        print(f"  Average test coverage: {avg_coverage:.3f}")
        print(f"  {1-delta:.2f}% test coverage: {quant_coverage:.3f}")
        print(f"  Average shift coverage: {avg_shift_coverage:.3f}")
        print(f"  {1-delta:.2f}% shift coverage: {quant_shift_coverage:.3f}")
        print(f"  Average interval size: {avg_size:.3f}")
        print("="*60)
    
    plot_distribution_comparisons(rep_results, bound_types, delta, target_risk)

    avg_shift = np.mean(shifts)
    var_shift = np.var(shifts)
    print(f"  Average W2 shift: {avg_shift:.3f}")
    print(f"  Variance W2 shift: {var_shift:.3f}")
    

if __name__ == "__main__":
    main()
