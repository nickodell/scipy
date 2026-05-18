

import numpy as np

from .common import Benchmark, safe_import

with safe_import():
    from scipy.linalg._decomp_update import qr_insert
    qr_insert_single = qr_insert.__wrapped__


# Load data

class QrBench(Benchmark):
    def setup(self):
        data = np.load('/home/nodell/scipy/qr_prob.npz')
        self.Q2 = data["Q2"]
        self.R2 = data["R2"]
        self.hcur = data["hcur"]
        self.j = data["j"].item()
        self.saved_Q = data["Q"]
        self.saved_R = data["R"]
    def time_qr(self):
        Q, R = qr_insert(self.Q2, self.R2, self.hcur, self.j, which='col',
                         overwrite_qru=True, check_finite=False)
        #np.testing.assert_allclose(Q, self.saved_Q)
        #np.testing.assert_allclose(R, self.saved_R)
    def time_qr_single(self):
        Q, R = qr_insert_single(self.Q2, self.R2, self.hcur, self.j, which='col',
                                overwrite_qru=True, check_finite=False)
        #np.testing.assert_allclose(Q, self.saved_Q)
        #np.testing.assert_allclose(R, self.saved_R)

