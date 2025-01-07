import math
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
import sys
import gc
from LBFGS import FullBatchLBFGS
import numpy as np
import os
from tqdm import tqdm


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, device_ids, output_device, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.keops.MaternKernel(nu=2.5))

        self.covar_module = gpytorch.kernels.MultiDeviceKernel(
            base_covar_module, device_ids=device_ids,
            output_device=output_device
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class VariationalGP(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(VariationalGP, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class BasicGP(gpytorch.models.ExactGP):             # For CPU Usage
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_variational_gp(train_loader, train_x, train_y, output_device, num_epochs=20):

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(output_device)
    model = VariationalGP(train_x[:500, :]).to(output_device)   # 500 is number of inducing points
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=0.01)

    # Our loss object. We're using the VariationalELBO
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

    print("Training GP:")
    epochs_iter = tqdm(range(num_epochs), desc="Epoch")
    for i in epochs_iter:
        # Within each iteration, we will go over each minibatch of data
        minibatch_iter = tqdm(train_loader, desc="Minibatch:", leave=False)
        for x_batch, y_batch in minibatch_iter:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            minibatch_iter.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()

    return model, likelihood


def predict_variational_gp(test_loader, gp, likelihood, dango_list, embedding_function, output_device):
    gp.eval()
    likelihood.eval()
    means = torch.tensor([0.])
    variances = torch.tensor([0.])
    batch_size = 15000
    print("Predicting with GP:")
    with torch.no_grad():
        batch_iter = tqdm(test_loader, desc="Batch:")
        for x_batch, _ in batch_iter:
            gp_pred_x = embedding_function(dango_list, x_batch).to(output_device)
            for j in range(math.ceil(len(gp_pred_x) / batch_size)):
                x = gp_pred_x[j * batch_size:min((j + 1) * batch_size, len(gp_pred_x))]
                preds = gp(x)
                del x
                means = torch.cat([means, preds.mean.cpu()])
                variances = torch.cat([variances, preds.variance.cpu()])
                del preds
            del gp_pred_x
    means = means[1:].cpu().numpy()
    variances = variances[1:].cpu().numpy()
    return means, variances


def train_gp(train_x,
          train_y,
          device_ids,
          output_device,
          checkpoint_size,
          preconditioner_size,
          n_training_iter,
):
    print(output_device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(output_device)
    model = ExactGPModel(train_x, train_y, device_ids, output_device, likelihood).to(output_device)
    model.train()
    likelihood.train()

    optimizer = FullBatchLBFGS(model.parameters(), lr=0.1)
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    with gpytorch.beta_features.checkpoint_kernel(checkpoint_size), \
         gpytorch.settings.max_preconditioner_size(preconditioner_size):

        def closure():
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            return loss

        loss = closure()
        loss.backward()

        for i in range(n_training_iter):
            options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
            loss, _, _, _, _, _, _, fail = optimizer.step(options)

            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, n_training_iter, loss.item(),
                model.covar_module.module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()
            ))

            if fail:
                print('Convergence reached!')
                break

    print(f"Finished training Gaussian Process on {train_x.size(0)} data points using {len(device_ids)} GPUs.")
    return model, likelihood


def train_gp_cpu(model, likelihood, train_x, train_y, training_iter):
    model.train()
    likelihood.train()
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()


def predict_gp(GP, pred_x, devices):
    pred_x = pred_x.to('cpu')
    batch_size = 100
    pred_means = []
    pred_vars = []
    GP.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(35):
        for j in range(math.ceil(len(pred_x) / batch_size)):
            x = pred_x[j * batch_size:min((j + 1) * batch_size, len(pred_x))].to(devices[0])
            print(x.shape)
            torch.cuda.memory_summary(device=None, abbreviated=False)
            prediction = GP(x)
            mean = prediction.mean.detach().cpu().numpy()
            var = prediction.variance.detach().cpu().numpy()
            pred_means.append(mean)
            pred_vars.append(var)
        pred_means = torch.cat(pred_means, dim=0)
        pred_vars = torch.cat(pred_vars, dim=0)
    pred_vars = pred_vars.detach().cpu().numpy()
    pred_means = pred_means.detach().cpu().numpy()
    print(pred_means)
    print(pred_means.flatten())
    print(pred_vars.flatten())
    return pred_vars, pred_means



def get_free_gpus(numGPUs):
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > ./tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    if len(memory_available) > 0:
        max_mem = np.max(memory_available)
        print(max_mem)
        ids = np.where(memory_available == max_mem)[0]
        print(ids)
        chosen_ids = np.random.choice(ids, numGPUs)
        print("setting to gpu:%d" % chosen_ids[0])
        torch.cuda.set_device("cuda:%d"%chosen_ids[0])
        return chosen_ids
    else:
        return


def find_best_gpu_setting(train_x,
                          train_y,
                          n_devices,
                          output_device,
                          preconditioner_size):
    N = train_x.size(0)

    # Find the optimum partition/checkpoint size by decreasing in powers of 2
    # Start with no partitioning (size = 0)
    settings = [0] + [int(n) for n in np.ceil(N / 2**np.arange(1, np.floor(np.log2(N))))]

    for checkpoint_size in settings:
        print('Number of devices: {} -- Kernel partition size: {}'.format(n_devices, checkpoint_size))
        try:
            # Try a full forward and backward pass with this setting to check memory usage
            _, _ = train_gp(train_x, train_y,
                         device_ids=n_devices, output_device=output_device,
                         checkpoint_size=checkpoint_size,
                         preconditioner_size=preconditioner_size, n_training_iter=1)

            # when successful, break out of for-loop and jump to finally block
            break
        except RuntimeError as e:
            print('RuntimeError: {}'.format(e))
        except AttributeError as e:
            print('AttributeError: {}'.format(e))
        finally:
            # handle CUDA OOM error
            gc.collect()
            torch.cuda.empty_cache()
    return checkpoint_size


# if __name__ == '__main__':
#     n_devices = torch.cuda.device_count()
#     print('Planning to run on {} GPUs.'.format(n_devices))
#     preconditioner_size = 100
#     # checkpoint_size = find_best_gpu_setting(train_x, train_y,
#     #                                         n_devices=n_devices,
#     #                                         output_device=output_device,
#     #                                         preconditioner_size=preconditioner_size)
#     model, likelihood = trainGP(train_x, train_y,
#                               n_devices=2, output_device=output_device,
#                               checkpoint_size=10000,
#                               preconditioner_size=100,
#                               n_training_iter=20)
#     print(model)