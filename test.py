import mxnet as mx
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)
import os, sys
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../../tests/python/gpu'))
import test_operator_gpu as gpu
from scipy.spatial.distance import cdist
from numpy.testing import assert_allclose

def test_permutohedral_with_shape(data_shape, pos_shape):
    sym = mx.sym.Permutohedral(name='perm')
    cpu_ctx = {'ctx': mx.cpu(0), 'perm_data': data_shape, 'perm_pos': pos_shape}
    gpu_ctx = {'ctx': mx.gpu(0), 'perm_data': data_shape, 'perm_pos': pos_shape}
    ctx_list = [cpu_ctx, gpu_ctx]
    gpu.check_consistency(sym, ctx_list, scale=50)

    print gpu.check_speed(sym, cpu_ctx, scale=50, N=10)
    print gpu.check_speed(sym, gpu_ctx, scale=50, N=10)

def cpu_permuto(data, pos, norm=False):
    out = []
    for i in range(data.shape[0]):
        x = data[i]
        x = x.reshape((x.shape[0], x.size/x.shape[0])).T
        if norm:
            x = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
        p = pos[i]
        p = p.reshape((p.shape[0], p.size/p.shape[0])).T
        kernel = np.exp(-0.5*cdist(p, p, metric='euclidean')**2)
        y = kernel.dot(x)
        if norm:
            y = y[:, :-1] / y[:, -1:]
        out.append(y.T)
    out = np.asarray(out).reshape(data.shape)
    return out

def forward(data, pos, norm=False):
    sym = mx.sym.Permutohedral(data=mx.sym.Variable('val'), pos=mx.sym.Variable('pos'), normalize=norm)
    exe = sym.simple_bind(grad_req='write', ctx=mx.gpu(0), val=data.shape, pos=pos.shape)
    exe.arg_arrays[0][:] = data
    exe.arg_arrays[1][:] = pos
    exe.forward(is_train=False)
    return exe.outputs[0].asnumpy()

def cpu_gradient(data, pos, ograd):
    shape = list(data.shape)
    shape[1] = 1
    exdata = np.append(data, np.ones(shape), axis=1)

    exout = cpu_permuto(exdata, pos, norm=False)
    gexout = forward(exdata, pos, norm=False)
    print 'exout', np.abs((gexout - exout)/exout)
    norm = exout[:, -1:]
    gnorm = gexout[:, -1:]
    out = exout[:, :-1]
    gout = gexout[:, :-1]

    exograd = np.append(ograd/norm, -(out*ograd/(norm**2)).sum(axis=1, keepdims=True), axis=1)
    gexograd = np.append(ograd/gnorm, -(gout*ograd/(gnorm**2)).sum(axis=1, keepdims=True), axis=1)
    print 'exograd', np.abs((gexograd - exograd)/exograd)

    pgrad = np.zeros(pos.shape)
    gpgrad = np.zeros(pos.shape)

    for i in range(exdata.shape[1]):
        f1 = exograd[:, i:i+1]*cpu_permuto(pos*exdata[:, i:i+1], pos, norm=False)
        f2 = exograd[:, i:i+1]*pos*cpu_permuto(exdata[:, i:i+1], pos, norm=False)
        f3 = exdata[:, i:i+1]*cpu_permuto(exograd[:, i:i+1]*pos, pos, norm=False)
        f4 = exdata[:, i:i+1]*pos*cpu_permuto(exograd[:, i:i+1], pos, norm=False)

        gf1 = gexograd[:, i:i+1]*forward(pos*exdata[:, i:i+1], pos, norm=False)
        gf2 = gexograd[:, i:i+1]*pos*forward(exdata[:, i:i+1], pos, norm=False)
        gf3 = exdata[:, i:i+1]*forward(gexograd[:, i:i+1]*pos, pos, norm=False)
        gf4 = exdata[:, i:i+1]*pos*forward(gexograd[:, i:i+1], pos, norm=False)

        pgrad += f1 - f2 + f3 - f4
        gpgrad += gf1 - gf2 + gf3 - gf4
        tt = f1-f2
        gtt = gf1 - gf2
        print 'pgrad', np.abs((gpgrad - pgrad)/pgrad)

    return pgrad

def check_gradient(sym, ctx, eps, scale):
    exe = sym.simple_bind(grad_req='write', **ctx)
    for w in exe.arg_arrays:
        w[:] = np.random.uniform(-scale, scale, size=w.shape)
    exe.arg_arrays[1][:] = np.ceil(np.random.uniform(-1, 1, size=w.shape))*100
    og = mx.nd.ones(shape=exe.outputs[0].shape, ctx=exe.outputs[0].context)
    exe.forward(is_train=True)
    exe.backward(og)
    grad = [g.asnumpy() for g in exe.grad_arrays]

    for w, g in zip(exe.arg_arrays, grad):
        npw = w.asnumpy()
        for i in range(npw.size):
            npw.flat[i] -= eps
            w[:] = npw
            exe.forward(is_train=True)
            f0 = exe.outputs[0].asnumpy()

            npw.flat[i] += 2*eps
            w[:] = npw
            exe.forward(is_train=True)
            f1 = exe.outputs[0].asnumpy()

            approx = (f1-f0).sum()/(2*eps)

            print g.flat[i], approx

            npw.flat[i] -= eps

def check_gradient2(sym, ctx, eps, scale, norm=False):
    exe = sym.simple_bind(grad_req='write', **ctx)
    for w in exe.arg_arrays:
        w[:] = np.random.uniform(-scale, scale, size=w.shape)
    exe.arg_arrays[1][:] = np.random.uniform(-2, 2, size=w.shape)
    og = mx.nd.ones(shape=exe.outputs[0].shape, ctx=exe.outputs[0].context)
    exe.forward(is_train=True)
    data, pos = [arr.asnumpy() for arr in exe.arg_arrays]
    #assert_allclose(exe.outputs[0].asnumpy(), cpu_permuto(data, pos, True), rtol=1e-3, atol=1e-5)
    #exe.outputs[0][:] = cpu_permuto(data, pos, True)
    exe.backward(og)
    grad = [g.asnumpy() for g in exe.grad_arrays]
    #grad[1] = cpu_gradient(data, pos, og.asnumpy())

    stat = []
    sign = []
    for w, g in zip(exe.arg_arrays, grad):
        npw = w.asnumpy()
        for i in range(npw.size):
            npw.flat[i] -= eps
            w[:] = npw
            exe.forward(is_train=True)
            data, pos = [arr.asnumpy() for arr in exe.arg_arrays]
            f0cpu = cpu_permuto(data, pos, norm=norm)
            f0 = exe.outputs[0].asnumpy()

            npw.flat[i] += 2*eps
            w[:] = npw
            exe.forward(is_train=True)
            data, pos = [arr.asnumpy() for arr in exe.arg_arrays]
            f1cpu = cpu_permuto(data, pos, norm=norm)
            f1 = exe.outputs[0].asnumpy()

            approx = (f1-f0).sum()/(2*eps)
            approxcpu = approx# (f1cpu-f0cpu).sum()/(2*eps)

            print np.abs((g.flat[i] - approxcpu)/(approxcpu + 1e-8)), g.flat[i], approxcpu
            stat.append(np.abs((g.flat[i] - approxcpu)/(approxcpu + 1e-8)))
            sign.append(1 if g.flat[i]*approxcpu > 0 else 0)

            npw.flat[i] -= eps

    print np.histogram(stat, range=(np.percentile(stat, 5), np.percentile(stat, 95)))
    print np.bincount(sign)

# test_permutohedral_with_shape((1, 3, 255, 255), (1, 5, 255, 255))

# sym = mx.sym.Permutohedral(data=mx.sym.Variable('val'), pos=mx.sym.Variable('pos'), normalize=True)
# exe = sym.simple_bind(ctx=mx.gpu(0), val=(1,1,20,20), pos=(1,5,20,20))

# exe.arg_arrays[0][:] = np.random.uniform(-1, 1, size=exe.arg_arrays[0].shape)
# exe.arg_arrays[1][:] = np.random.uniform(-5, 5, size=exe.arg_arrays[1].shape)
# exe.forward()

# o1 =  exe.outputs[0].asnumpy()
# o2 = cpu_permuto(exe.arg_arrays[0].asnumpy(), exe.arg_arrays[1].asnumpy(), True)
# print o1
# print ' '
# print o2
# print ' '
# print o1/o2
# print (o1/o2).std()/(o1/o2).mean()
norm = True
sym = mx.sym.Permutohedral(data=mx.sym.Variable('val'), pos=mx.sym.Variable('pos'), normalize=norm)
ctx = {'ctx': mx.gpu(0), 'val': (1,2,5,5), 'pos': (1,2,5,5)}

check_gradient2(sym, ctx, 1e-3, 5, norm)
