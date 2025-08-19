r"""
Polynomial regression
=====================

Polynomial regression is a simple 1D regression. The samples are of the form :math:`\xi = (x,y) \in \mathbb{R}\times\mathbb{R}` and the sought predictor is of the form  :math:`f(x) = \sum_{i=0}^d a_i x^i` where :math:`(a_0,..,a_d)` are the :math:`d+1` coefficients to lean.

In the following example, we seek to learn a polynomial fitting the function 

.. math::

    f^\star(x) = \frac{10}{e^{x}+e^{-x}} + x

from :math:`n=100` samples uniformly drawn from :math:`[-2,2]` and corrupted by a Gaussian noise with zero mean and variance :math:`0.1`. 


"""
from typing import Iterable
import numpy as np
import matplotlib.pyplot as plt
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


from skwdro.torch import robustify
from skwdro.solvers.oracle_torch import DualLoss



pt.manual_seed(42)
np.random.seed(42)

folder = "./Learned_WDRO_Out/"
plt.style.use("paper.mplstyle")

# %%
# Problem setup
# ~~~~~~~~~~~~~

n = 250                                     # Number of observations

degree = 4                                 # Degree of the regression

radius = pt.tensor(np.sqrt(2.0)/np.sqrt(n))                     # Robustness radius

var = pt.tensor(1.0)                        # Variance of the noise

def f_star(x):                              # Generating function
    return 10/(pt.exp(x)+pt.exp(-x)) + x
 
xi = pt.rand(n)*4.0 - 2.0                   # x_i's are uniformly drawn from (-2,2] 
xi = pt.sort(xi)[0]                         # we sort them for easier plotting
yi = f_star(xi) + pt.sqrt(var)*pt.randn(n)  # y_i's are f(x_i) + noise
xi = xi.unsqueeze(-1)
yi = yi.unsqueeze(-1)


# print(f"{xi.shape=},{yi.shape=}")

dataset = DataLoader(TensorDataset(xi, yi), batch_size=n, shuffle=True)



device = "cuda" if pt.cuda.is_available() else "cpu"



# testing data
n_test = 1000
xi_test = pt.rand(n_test)*4.0 - 2.0                   # x_i's are uniformly drawn from (-2,2] 
xi_test = pt.sort(xi_test)[0]                         # we sort them for easier plotting
yi_test = f_star(xi_test) + pt.sqrt(var)*pt.randn(n_test)  # y_i's are f(x_i) + noise
xi_test = xi_test.unsqueeze(-1)
yi_test = yi_test.unsqueeze(-1)


def coverage(model,lam):
    cov = (pt.abs(model(xi_test[:,None,None]).detach().cpu().squeeze() - yi_test) <= lam ).float().mean()
    return 1. - cov.item()

# %%
# Scikit-learn
# ~~~~~~~~~~~~

# Train the polynomial regression model using sklearn
xi_np = xi.numpy().reshape(-1, 1)  # Convert xi to numpy and reshape
yi_np = yi.numpy()


poly = PolynomialFeatures(degree)
xi_poly = poly.fit_transform(xi_np)  # Transform xi into polynomial features


# Fit the linear regression model
model_sklearn = LinearRegression(fit_intercept=False)
model_sklearn.fit(xi_poly, yi_np)


tosave =  np.vstack((xi_np.squeeze(),yi_np.squeeze())).T
np.savetxt(folder+"data.csv", tosave , delimiter=',', fmt='%f', header="x , y ")

# %%
# Polynomial model
# ~~~~~~~~~~~~~~~~

class PolynomialModel(nn.Module):
    def __init__(self, degree : int) -> None:
        super().__init__()
        self._degree = degree
        self.linear = nn.Linear(self._degree+1, 1, bias=False)

    def _polynomial_features(self, x):
        return pt.cat([x ** i for i in range(self._degree + 1)],dim=-1)

    def forward(self, x):
        return self.linear(self._polynomial_features(x))



# %%
# Polynomial model w/ fixed weights
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class PolynomialFixedModel(nn.Module):
    def __init__(self, degree : int, weights) -> None:
        super().__init__()
        self._degree = degree
        self.linear = nn.Linear(self._degree+1, 1, bias=False)
        self.linear.weight.requires_grad = False

        # Set the weights and bias
        with pt.no_grad():
            self.linear.weight.copy_(pt.tensor(weights).reshape(1,-1))  # Shape: (1, d)
            #print(f"{self.linear.weight = }")

            # Disable gradient computation for weights and bias
        self.linear.weight.requires_grad = False

    def _polynomial_features(self, x):
        return pt.cat([x ** i for i in range(self._degree + 1)],dim=-1)

    def forward(self, x):
        self.linear.weight.requires_grad = False
        #print(f"{self.linear.weight = }")
        return self.linear(self._polynomial_features(x))



# %%
# Training loop
# ~~~~~~~~~~~~~

def train(dual_loss: DualLoss, dataset: Iterable[tuple[pt.Tensor, pt.Tensor]], epochs: int=50):

    lbfgs = pt.optim.LBFGS(dual_loss.parameters(),lr=1.,line_search_fn="strong_wolfe")   # LBFGS is used to optimize thanks to the nature of the problem

    def closure():          # Closure for the LBFGS solver
        lbfgs.zero_grad()
        loss = dual_loss(xi, xi_label, reset_sampler=True).mean()
        loss.backward()
        return loss

    pbar = tqdm(range(epochs))

    for _ in pbar:
        # Every now and then, try to rectify the dual parameter (e.g. once per epoch).
        dual_loss.get_initial_guess_at_dual(*next(iter(dataset))) # *
        if dual_loss._lam < 0.:
            with pt.no_grad():
                dual_loss._lam *= 0.

        # Main train loop
        inpbar = tqdm(dataset, leave=False)
        for xi, xi_label in inpbar:
            lbfgs.step(closure)
            if dual_loss._lam < 0.:
                with pt.no_grad():
                    dual_loss._lam *= 0.
                    dual_loss._lam.requires_grad = False

        pbar.set_postfix({"lambda": f"{dual_loss.lam.item():.2f}"})

    return dual_loss

# %%
# Training loop 2
# ~~~~~~~~~~~~~~~

def train2(dual_loss: DualLoss, dataset: Iterable[tuple[pt.Tensor, pt.Tensor]], epochs: int=50):

    #optimizer = pt.optim.SGD(params=dual_loss.parameters())
    optimizer = pt.optim.AdamW(params=dual_loss.parameters(),lr=0.1)

    # Training loop
    iterator = tqdm(range(epochs), position=0, desc='Epochs', leave=False)
    losses = []
    for epoch in iterator:

        for batch_x, batch_y in tqdm(dataset, position=1, desc='Sample', leave=False):

            optimizer.zero_grad()
            loss = dual_loss(batch_x, batch_y, reset_sampler=True)
            loss.backward()
            optimizer.step()
            if dual_loss.lam < 0.:
                with pt.no_grad():
                    dual_loss._lam *= 0.



    return dual_loss

# %%
# Training function
# ~~~~~~~~~~~~~~~~~


def trainConfModel(model, dataset: Iterable[tuple[pt.Tensor, pt.Tensor]], predsize  = pt.tensor(1.0), radius =  pt.tensor(1.0), epochs: int=50):

    def conformal_loss(output : pt.Tensor, target: pt.Tensor) -> pt.Tensor :

        # ## Option 1
        # loss = F.l1_loss(output,target,reduction='none')> predsize
        # loss = loss.float()

        # ## Option 2
        # loss =  F.relu(F.l1_loss(output,target,reduction='none') - predsize)

        ## Option 3
        loss = F.sigmoid( 30.0*(F.l1_loss(output,target,reduction='none') - predsize) + 3 ) # Boosted sigmoid

        return loss.squeeze(-1)

    rob_model = robustify( 
                conformal_loss,
                model,
                radius,
                xi,
                yi
            ) # Replaces the loss of the model by the dual WDRO loss

    # trained_model = train(rob_model, dataset, epochs=epochs) # type: ignore
    trained_model = train2(rob_model, dataset, epochs=epochs) # type: ignore

    trained_model.eval()  # type: ignore
    
    return trained_model

# %%
# Loop on predsizes
# ~~~~~~~~~~~~~~~~~

predsize_list = np.linspace(1,5,21)

loss_list = []
loss_fixed_list = []
cov_list = []
cov_fixed_list = []

for Lambda in predsize_list:

    Lambda  = float(Lambda)

    predsize  = pt.tensor(Lambda)

    print(f"\n#########\n{Lambda=}\n#########\n")

    # %%
    # Training 
    # ~~~~~~~~


    model = PolynomialModel(degree).to(device)  # Our polynomial regression model
    trained_model = trainConfModel(model, dataset,  predsize  = predsize, epochs= 1000)
    wdro_loss = trained_model(xi,yi, reset_sampler=True).item()
    cov = coverage(trained_model.primal_loss.transform,predsize)
    print(f"{wdro_loss=} , {cov=}")
    loss_list.append(wdro_loss)
    cov_list.append(cov)

    model_fixed = PolynomialFixedModel(degree,weights= model_sklearn.coef_.reshape(-1,1)).to(device)  # Our polynomial regression model
    trained_model_fixed = trainConfModel(model_fixed, dataset,  predsize  = predsize , epochs= 100)
    wdro_loss_fixed = trained_model_fixed(xi,yi, reset_sampler=True).item()
    cov_fixed = coverage(trained_model_fixed.primal_loss.transform,predsize)
    print(f"{wdro_loss_fixed=} , {cov_fixed=}")
    loss_fixed_list.append(wdro_loss_fixed)
    cov_fixed_list.append(cov_fixed)


    # %%
    # Results
    # ~~~~~~~
    #
    # We plot the obtained polynomial and print the coefficients

    fig, ax = plt.subplots()
    xtrial = pt.linspace(-2.1,2.1,100)
    ax.scatter(xi.cpu(), yi.cpu(), c='g', label='train data')
    ax.plot(xtrial, f_star(xtrial), 'k', label='generating function')

    # predsk = model_sklearn.predict(poly.transform(xtrial.numpy().reshape(-1,1)))  # Transform xi into polynomial features
    # ax.plot(xtrial,predsk, 'g', label='Sklearn prediction')  

    pred2 = trained_model_fixed.primal_loss.transform(xtrial[:,None,None]).detach().cpu().squeeze() # type: ignore
    ax.plot(xtrial,pred2, 'b', label='Separately trained polynomial model')  


    pred1 = trained_model.primal_loss.transform(xtrial[:,None,None]).detach().cpu().squeeze() # type: ignore
    ax.plot(xtrial,pred1, 'r', label='Jointly trained polynomial model')  

    plt.xlim([-2,2])
    plt.ylim([-2,8])
    fig.legend()
    # plt.show()
    plt.savefig(folder+f"models_{Lambda=}.pdf")
    plt.close()

    tosave =  np.vstack((xtrial,f_star(xtrial),pred1,pred2)).T
    np.savetxt(folder+f"models_{Lambda=}.csv", tosave , delimiter=',', fmt='%f',header=" x , f_star(x) , pred_trained, pred_fixed ")


    # coeffs = model_sklearn.coef_.squeeze()
    # polyString = "Sklearn Polynomial regressor  (degree {:d})\n".format(degree)
    # for i,a in enumerate(coeffs):
    #     if i>0:
    #         if a>=0.0:
    #             polyString += "+ {:3.2f}x**{:d} ".format(a,i)
    #         else:
    #             polyString += "- {:3.2f}x**{:d} ".format(abs(a),i)
    #     else:
    #         if a>=0.0:
    #             polyString += "{:3.2f} ".format(a)
    #         else:
    #             polyString += "- {:3.2f} ".format(abs(a))
    # print(polyString)




    coeffs = trained_model.primal_loss.transform.linear.weight.tolist()[0] # type: ignore
    polyString = "Polynomial regressor for conformal wdro loss (degree {:d}, radius {:3.2e}):\n ".format(degree,radius.float())
    for i,a in enumerate(coeffs):
        if i>0:
            if a>=0.0:
                polyString += "+ {:3.2f}x**{:d} ".format(a,i)
            else:
                polyString += "- {:3.2f}x**{:d} ".format(abs(a),i)
        else:
            if a>=0.0:
                polyString += "{:3.2f} ".format(a)
            else:
                polyString += "- {:3.2f} ".format(abs(a))

    print(polyString)




    coeffs = trained_model_fixed.primal_loss.transform.linear.weight.tolist()[0] # type: ignore
    polyString = "Polynomial regressor (degree {:d}, radius {:3.2e}):\n  ".format(degree,radius.float())
    for i,a in enumerate(coeffs):
        if i>0:
            if a>=0.0:
                polyString += "+ {:3.2f}x**{:d} ".format(a,i)
            else:
                polyString += "- {:3.2f}x**{:d} ".format(abs(a),i)
        else:
            if a>=0.0:
                polyString += "{:3.2f} ".format(a)
            else:
                polyString += "- {:3.2f} ".format(abs(a))

    print(polyString)


tosave =  np.vstack((predsize_list,loss_list,cov_list,loss_fixed_list,cov_fixed_list)).T
np.savetxt(folder+"learned_WDRO.csv", tosave , delimiter=',', fmt='%f', header="lambda, loss w/ training, risk w/ training, loss w/o training, risk w/o training")

fig, ax = plt.subplots()
ax.plot(predsize_list,loss_fixed_list, 'b', label='Separate: bound') 
ax.plot(predsize_list,cov_fixed_list, 'b--' , label='Separate: risk')   
ax.plot(predsize_list,loss_list, 'r', label='Joint: bound') 
ax.plot(predsize_list,cov_list, 'r--' , label='Joint: risk')   
plt.xlabel("$\lambda$")
fig.legend()
# plt.show()
plt.savefig(folder+f"Loss_Risk.pdf")
