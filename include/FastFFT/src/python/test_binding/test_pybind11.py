
import fastfft_test as f
from cupy import array as cp_array
from cupy import float32 as fp32

from torch import tensor as t_tensor
from torch import device as t_device
from torch import float32 as t_fp32
from torch import numel as t_numel

print('Entering test pybin11')

# The template specialization is encoded in the binding name
t = f.TestClass_float_float(3.4,3)

# Checking that the construction worked out.
print('Member variable one has value {}'.format(t.getOne()))


# Make a float array on the device.
# should this be ascontiguousarray?
a = cp_array([1,2,3], dtype=fp32)

#
print('Array, before {}'.format(a))

# Pass in the pointer (which must be cast in cuda, and also the array length.)
# TODO: Also consider padding.
t.sum_cupy_array(a.data.ptr, a.__len__())
# The output is the reduced sum in the first element of the array.
print('Output of the cupy reduce functions {}'.format(a))

# Testing torch
cuda0 = t_device('cuda:0')
a = t_tensor([3, 2, 4], dtype=t_fp32, device=cuda0)

print('Repeating the test but from a torch tensor')
print('Array, before {}'.format(a))
t.sum_cupy_array(a.data_ptr(), t_numel(a))
print('Output of the cupy reduce functions {}'.format(a))
