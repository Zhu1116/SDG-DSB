##########################
#    copy from HySUPP    #
##########################
import scipy.io as sio
import numpy as np
import spams


class AdditiveWhiteGaussianNoise:
    def __init__(self, SNR=None):
        self.SNR = SNR

    def apply(self, Y):
        """
        Compute sigmas for the desired SNR given a flattened input HSI Y
        """
        assert len(Y.shape) == 2
        L, N = Y.shape

        #######
        # Fit #
        #######
        if self.SNR is None:
            sigmas = np.zeros(L)
        else:
            assert self.SNR > 0, "SNR must be strictly positive"
            # Uniform across bands
            sigmas = np.ones(L)
            # Normalization
            sigmas /= np.linalg.norm(sigmas)
            # Compute sigma mean based on SNR
            num = np.sum(Y**2) / N
            denom = 10 ** (self.SNR / 10)
            sigmas_mean = np.sqrt(num / denom)
            # Noise variance
            sigmas *= sigmas_mean

        #############
        # Transform #
        #############
        noise = np.diag(sigmas) @ np.random.randn(L, N)

        # Return additive noise
        return Y + noise


class ArchetypalAnalysis():
    def __init__(
        self,
        epsilon=1e-3,
        robust=False,
        computeXtX=True,
        stepsFISTA=3,
        stepsAS=100,
        randominit=False,
        numThreads=-1,
        *args,
        **kwargs,
    ):

        super().__init__()
        self.params = {
            "epsilon": epsilon,
            "robust": robust,
            "computeXtX": computeXtX,
            "stepsFISTA": stepsFISTA,
            "stepsAS": stepsAS,
            "randominit": randominit,
            "numThreads": numThreads,
        }

    def compute_endmembers_and_abundances(
        self,
        Y,
        p,
        *args,
        **kwargs,
    ):
        """
        Archetypal Analysis optimizer from SPAMS

        Parameters:
            Y: `numpy array`
                2D data matrix (L x N)

            p: `int`
                Number of endmembers

            E0: `numpy array`
                2D initial endmember matrix (L x p)
                Default: None

        Source: http://thoth.inrialpes.fr/people/mairal/spams/doc-python/html/doc_spams004.html#sec8
        """
        Yf = np.asfortranarray(Y, dtype=np.float64)

        Ehat, Asparse, Bsparse = spams.archetypalAnalysis(
            Yf,
            p=p,
            returnAB=True,
            **self.params,
        )

        Ahat = np.array(Asparse.todense())
        self.B = np.array(Bsparse.todense())

        return Ehat, Ahat


def unmix(data, p=6):
    # Get noise
    noise = AdditiveWhiteGaussianNoise()

    H, W = data.shape[0], data.shape[1]
    Y = data.reshape(H * W, -1)
    Y = Y.transpose()

    # Apply noise
    Y = noise.apply(Y)
    # Build model
    model = ArchetypalAnalysis()
    # Solve unmixing
    E_hat, A_hat = model.compute_endmembers_and_abundances(
        Y,
        p,
        H=H,
        W=W,
    )

    return E_hat, A_hat
