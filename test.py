import mxnet as mx
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)
import os, sys
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../../tests/python/gpu'))
import test_operator_gpu as gpu
from scipy.spatial.distance import cdist

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

#test_permutohedral_with_shape((1, 3, 255, 255), (1, 5, 255, 255))

sym = mx.sym.Permutohedral(data=mx.sym.Variable('val'), pos=mx.sym.Variable('pos'), normalize=True)
exe = sym.simple_bind(ctx=mx.gpu(0), val=(1,1,20,20), pos=(1,5,20,20))

exe.arg_arrays[0][:] = np.random.uniform(-1, 1, size=exe.arg_arrays[0].shape)
exe.arg_arrays[1][:] = np.random.uniform(-5, 5, size=exe.arg_arrays[1].shape)
exe.forward()

o1 =  exe.outputs[0].asnumpy()
o2 = cpu_permuto(exe.arg_arrays[0].asnumpy(), exe.arg_arrays[1].asnumpy(), True)
print o1
print ' '
print o2
print ' '
print o1/o2
print (o1/o2).std()/(o1/o2).mean()
