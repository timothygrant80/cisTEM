import cupy as cp
import FastFFT 

FT =  FastFFT.FourierTransformer_float_float_float()

# Adding in the FFTW padding on the x-axis... logical dim is 16 16
a = cp.zeros([16,18],dtype=cp.float32)
# Normally I would set this at the origin, but I want to see if the padding is incorrect.
a[0][0] = 1.0

print("Cupy array is {}".format(a))

# Setup the plans
FT.
(16,16,1,16,16,1, True)
FT.Wait()
FT.SetInverseFFTPlan(16,16,1,16,16,1, True)
FT.Wait()
FT.SetInputPointerFromPython(a.data.ptr)
FT.Wait()
# These bools are defaults that are deprecated
FT.FwdFFT(False, True)
FT.Wait()
FT.InvFFT(True)

# Synchronize the default perthread stream
FT.Wait()
print("Cupy array is {}".format(a))