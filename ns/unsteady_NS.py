import os
import sys
import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pathlib
import paddle
import paddle.nn as nn
import paddle.static as static
from pyDOE import lhs
from mpl_toolkits.axes_grid1 import make_axes_locatable
from basic_model import DeepModelSingle, DeepModelMulti
from parser_pinn import get_parser
from gen_ns_data import get_noise_data, get_truth

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # CPU:-1; GPU0: 1; GPU1: 0;

parser_PINN = get_parser()
args = parser_PINN.parse_args()
path = pathlib.Path(args.save_path)
path.mkdir(exist_ok=True, parents=True)
for key, val in vars(args).items():
    print(f"{key} = {val}")
with open(path.joinpath('config'), 'wt') as f:
    f.writelines([f"{key} = {val}\n" for key, val in vars(args).items()])
adam_iter: int = int(args.adam_iter)
bfgs_iter: int = int(args.bfgs_iter)
verbose: bool = bool(args.verbose)
repeat: int = int(args.repeat)
start_epoch: int = int(args.start_epoch)
num_neurons: int = int(args.num_neurons)
num_layers: int = int(args.num_layers)


class PINN_NS_unsteady(DeepModelSingle):
    def __init__(self, planes):
        super(PINN_NS_unsteady, self).__init__(planes, active=nn.Tanh())
        self.lambda_1 = paddle.create_parameter(shape=[1], dtype='float32')
        self.lambda_2 = paddle.create_parameter(shape=[1], dtype='float32')

    def gradients(self, y, x):
        return paddle.grad(y, x, grad_outputs=paddle.ones_like(y),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    def get_lambda_1(self):
        return paddle.to_tensor(self.lambda_1)

    def get_lambda_2(self):
        return paddle.to_tensor(self.lambda_2)

    def equation(self, inn_var):
        model = self
        out_var = self.forward(inn_var)
        psi = out_var[..., 0:1]
        p = out_var[..., 1:2]

        dpsida = paddle.incubate.autograd.grad(psi, inn_var)
        u, v = dpsida[:, 1:2], -dpsida[:, 0:1]

        duda = paddle.incubate.autograd.grad(u, inn_var)
        dvda = paddle.incubate.autograd.grad(v, inn_var)
        dpda = paddle.incubate.autograd.grad(p, inn_var)

        dudx, dudy, dudt = duda[..., 0:1], duda[..., 1:2], duda[..., 2:3]
        dvdx, dvdy, dvdt = dvda[..., 0:1], dvda[..., 1:2], dvda[..., 2:3]
        dpdx, dpdy, dpdt = dpda[..., 0:1], dpda[..., 1:2], dpda[..., 2:3]

        d2udx2 = paddle.incubate.autograd.grad(dudx, inn_var)[..., 0:1]
        d2udy2 = paddle.incubate.autograd.grad(dudy, inn_var)[..., 1:2]
        d2vdx2 = paddle.incubate.autograd.grad(dvdx, inn_var)[..., 0:1]
        d2vdy2 = paddle.incubate.autograd.grad(dvdy, inn_var)[..., 1:2]

        res_u = dudt + (u * dudx + v * dudy) * self.get_lambda_1() + dpdx - (d2udx2 + d2udy2) * self.get_lambda_2()
        res_v = dvdt + (u * dvdx + v * dvdy) * self.get_lambda_1() + dpdy - (d2vdx2 + d2vdy2) * self.get_lambda_2()

        return res_u, res_v, p, u, v  # cat给定维度

    def predict_error(self, Val_pred):
        x, y, t, u, v, p = get_truth()
        out_pred = Val_pred.numpy()
        error_u = np.linalg.norm(out_pred[:, (1,)] - u, 2) / np.linalg.norm(u, 2)
        error_v = np.linalg.norm(out_pred[:, (2,)] - v, 2) / np.linalg.norm(v, 2)
        error_vel = np.sqrt(np.sum((out_pred[:, (1,)] - u) ** 2 + (out_pred[:, (2,)] - v) ** 2)) / np.sqrt(
            np.sum(u ** 2 + v ** 2))
        error_max = np.max(np.sqrt((out_pred[:, (1,)] - u) ** 2 + (out_pred[:, (2,)] - v) ** 2))
        error_p = np.linalg.norm(out_pred[:, (0,)] - p, 2) / np.linalg.norm(p, 2)
        lambda_1 = self.lambda_1.numpy()
        lambda_2 = self.lambda_2.numpy()
        error_lambda_1 = (np.abs((lambda_1 - 1.)))
        error_lambda_2 = (np.abs((lambda_2 - 0.01) / 0.01))

        return error_u, error_v, error_vel, error_max, error_p, error_lambda_1, error_lambda_2

    def plot_result(self, Val_pred, filename):
        x, y, t, u, v, p = get_truth()
        _shape = (50, 100)
        t_p = 100
        x = x[t_p::200, 0]
        y = y[t_p::200, 0]
        t = t[t_p::200, 0]
        u = u[t_p::200, 0]
        v = v[t_p::200, 0]
        p = p[t_p::200, 0]

        out_pred = Val_pred.numpy()
        plt.clf()
        fig, ax = plt.subplots(6, 1, figsize=(10, 20), dpi=200)
        fig.set_tight_layout(True)
        im = ax[0].pcolormesh(x.ravel().reshape(*_shape), y.ravel().reshape(*_shape),
                              out_pred[:, (1,)].ravel().reshape(*_shape), )
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        im = ax[1].pcolormesh(x.ravel().reshape(*_shape), y.ravel().reshape(*_shape), u.ravel().reshape(*_shape), )
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        im = ax[2].pcolormesh(x.ravel().reshape(*_shape), y.ravel().reshape(*_shape),
                              out_pred[:, (1,)].ravel().reshape(*_shape) - u.ravel().reshape(*_shape), cmap='bwr')
        divider = make_axes_locatable(ax[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        im = ax[3].pcolormesh(x.ravel().reshape(*_shape), y.ravel().reshape(*_shape),
                              out_pred[:, (0,)].ravel().reshape(*_shape), )
        divider = make_axes_locatable(ax[3])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        im = ax[4].pcolormesh(x.ravel().reshape(*_shape), y.ravel().reshape(*_shape), p.ravel().reshape(*_shape), )
        divider = make_axes_locatable(ax[4])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        im = ax[5].pcolormesh(x.ravel().reshape(*_shape), y.ravel().reshape(*_shape),
                              out_pred[:, (0,)].ravel().reshape(*_shape) - p.ravel().reshape(*_shape), cmap='bwr')
        divider = make_axes_locatable(ax[5])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax[5].set_title(f'mse: {np.mean(np.abs(out_pred[:, (0,)] - p)):.4f}')
        plt.colorbar(im, cax=cax)
        plt.savefig(filename)


def run_experiment(epoch_num, noise_type, noise, loss_type, weight, N=5000, _data=[], l_size=-1, abnormal_size=0):
    try:
        import paddle.fluid as fluid
        place = fluid.CUDAPlace(0) if paddle.is_compiled_with_cuda() else fluid.CPUPlace()
    except:
        place = None

    paddle.enable_static()
    paddle.incubate.autograd.enable_prim()

    ## 数据生成
    # Domain bounds
    lb = np.array([1, -2, 0])
    ub = np.array([8, 2, 20])

    if len(_data) == 0:
        x_train, y_train, t_train, u_train, v_train, p_train = get_noise_data(N=N, noise_type=noise_type, sigma=noise,
                                                                              size=abnormal_size)
        _data.append((x_train, y_train, t_train, u_train, v_train))
    x_train, y_train, t_train, u_train, v_train = _data[0]
    Tra_DATA = np.concatenate(_data[0], axis=1).astype('float32')

    # Visualize the collocation points
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    plt.scatter(_data[0][0].flatten(), _data[0][1].flatten(), marker='o', alpha=0.2, color='blue')
    plt.savefig('collocation.png')

    x, y, t, u, v, p = get_truth()
    Val_DATA = np.concatenate((x, y, t, u, v, p), axis=1).astype('float32')
    N_truth = Val_DATA.shape[0]

    ## 模型设置
    planes = [3] + [num_neurons] * num_layers + [2]
    # Model
    Net_model = PINN_NS_unsteady(planes=planes)
    # Loss
    # if loss_type == "square":
    #     Loss_data = nn.MSELoss()
    # elif loss_type == "l1":
    #     Loss_data = nn.L1Loss()
    # else:
    #     raise NotImplementedError(f'Loss type {loss_type} not implemented.')
    scheduler = paddle.optimizer.lr.MultiStepDecay(0.001, [adam_iter * 0.6, adam_iter * 0.8], gamma=0.1)
    Optimizer = paddle.optimizer.Adam(scheduler)

    Tra_inn = paddle.static.data('Tra_inn', shape=[x_train.shape[0], 3], dtype='float32')
    Tra_inn.stop_gradient = False
    Tra_out = paddle.static.data('Tra_out', shape=[u_train.shape[0], 2], dtype='float32')
    Tra_out.stop_gradient = False

    Val_inn = paddle.static.data('Val_inn', shape=[N_truth, 3], dtype='float32')
    res_u, res_v, p, u, v = Net_model.equation(Tra_inn)
    if loss_type == "square":
        u_loss = paddle.norm(u - Tra_out[:, 0], p=2) ** 2 / u.shape[0]
        v_loss = paddle.norm(v - Tra_out[:, 1], p=2) ** 2 / v.shape[0]
    elif loss_type == "l1":
        u_loss = paddle.norm(u - Tra_out[:, 0], p=1) / u.shape[0]
        v_loss = paddle.norm(v - Tra_out[:, 1], p=1) / v.shape[0]
    else:
        raise NotImplementedError(f'Loss type {loss_type} not implemented.')
    eqsU_loss = paddle.norm(res_u, p=2) ** 2 / res_u.shape[0]
    eqsV_loss = paddle.norm(res_v, p=2) ** 2 / res_v.shape[0]
    data_loss = u_loss + v_loss
    eqs_loss = eqsU_loss + eqsV_loss

    loss_batch = data_loss + weight * eqs_loss
    Optimizer.minimize(loss_batch)

    _, _, p_pre, u_pre, v_pre = Net_model.equation(Val_inn)
    Loss = [data_loss, eqs_loss, loss_batch]

    ## 执行训练过程
    log_loss = []
    print_freq = 20
    sta_time = time.time()

    exe = static.Executor(place)
    exe.run(static.default_startup_program())
    prog = static.default_main_program()
    start_epoch = 0

    for epoch in range(start_epoch, 1 + adam_iter):

        learning_rate = Optimizer.get_lr()
        exe.run(prog, feed={'Tra_inn': Tra_DATA[:, :3], 'Tra_out': Tra_DATA[:, 3:5], 'Val_inn': Val_DATA[:, :3]},
                fetch_list=[Loss[-1]])

        if epoch > 0 and epoch % print_freq == 0:
            all_items = exe.run(prog, feed={'Tra_inn': Tra_DATA[:, :3], 'Tra_out': Tra_DATA[:, 3:5],
                                            'Val_inn': Val_DATA[:, :3]},
                                fetch_list=[[p_pre, u_pre, v_pre] + Loss])
            p_pre = all_items[0]
            u_pre = all_items[1]
            v_pre = all_items[2]
            loss = all_items[3:]
            log_loss.append(np.array(loss).squeeze())
            print('epoch: {:6d}, lr: {:.3e}, data_loss: {:.3e}, pde_loss: {:.3e}, total_loss: {:.3e}, cost: {:.2f}'.
                  format(epoch, learning_rate, log_loss[-1][0], log_loss[-1][1], log_loss[-1][2],
                         time.time() - sta_time))
            paddle.save({'epoch': adam_iter, 'log_loss': log_loss,
                         'model': Net_model.state_dict(), "optimizer": Optimizer.state_dict()},
                        os.path.join(path,
                                     f'{epoch_num}_{loss_type}_{N}_{noise_type}_{noise}_{abnormal_size}_{weight}.pdparams'))

    paddle.save({'epoch': adam_iter, 'log_loss': log_loss,
                 'model': Net_model.state_dict(), "optimizer": Optimizer.state_dict()},
                os.path.join(path,
                             f'{epoch_num}_{loss_type}_{N}_{noise_type}_{noise}_{abnormal_size}_{weight}.pdparams'))
    error_u, error_v, error_vel, error_max, error_p, lambda_1_error, lambda_2_error = Net_model.predict_error(p_pre,
                                                                                                              u_pre,
                                                                                                              v_pre)
    Net_model.plot_result(p_pre, u_pre, v_pre, path.joinpath(
        f'{epoch_num}_{loss_type}_{N}_{noise_type}_{noise}_{abnormal_size}_{weight}.png'))
    with open(path.joinpath('result.csv'), 'a+') as f:
        f.write(
            f"{epoch_num},{loss_type},{N},{noise_type},{noise},{abnormal_size},{weight},{error_u},{error_v},{error_vel},{error_p},{lambda_1_error}, {lambda_2_error}\n")

    print("--- %s seconds ---" % (time.time() - sta_time))


def get_last_idx(filename):
    if not os.path.exists(filename):
        return -1
    with open(filename, "r") as f1:
        last_idx = int(f1.readlines()[-1].strip().split(',')[0])
        return last_idx