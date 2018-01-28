"""
Copyright (c) 2017, Florian Engels
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name "Framework on High-Resolution Frequency Estimation"
      nor the names of its contributors may be used to endorse or promote
      products derived from this software without specific prior written
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL FLORIAN ENGELS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from numpy import *

class FrameworkOnHighResolutionFrequencyEstimation:
    """ 
    A framework on computationally efficient, high-resolution frequency estimation

    Implementation of the framework proposed in

    [1] F. Engels, P. Heidenreich, A. Zoubir, F. Jondral, and M. Wintermantel, 
        "Advances in Automotive Radar: A framework on computationally efficient
         high-resolution frequency estimation",
        IEEE Signal Processing Magazine, Vol. 34, No. 2, Pp. 36-46.
   
    extended to process a fourth frequency dimension corresponding to elevation angle.

    Instance Variables 
    
    w_lambda,          -- window functions in range, velocity, and angular (azimuth/elevation) dimension, respectively
    w_mu,
    w_nu,
    w_omega
    Lf,Mf,Nf,Of        -- support of Fourier transform in range, velocity, and angular (azimuth/elevation) dimension, respectively
    L,M,N,O            -- support of peak neighborhood in range, velocity, and angular (azimuth/elevation) dimension, respectively
    T_db               -- threshold for multiple target indication based on the single-target fitting error  
    gamma_db           -- threshold for the generalized likelihood ratio test

    Ls,Ms,Ns,Os        -- sampling support in range, velocity, and angular (azimuth/elevation) dimension, respectively
    
    W_lambda,          -- model matrix in range, velocity, and angular (azimuth/elevation) dimension, respectively
    W_mu,
    W_nu,
    W_omega

    dfe                -- decoupled frequency estimation comprising K-target model fit in the resolution
                          dimension and subsequent single-target model fits in the remaining dimensions 

    """
    def __init__(self,
                 w_lambda,w_mu,w_nu,
                 Lf,Mf,Nf,L,M,N,
                 T_db,gamma_db,
                 w_omega=None,Of=1,O=1,
                 average=True):

        self.w_lambda,self.w_mu,self.w_nu = w_lambda,w_mu,w_nu
        self.Lf,self.Mf,self.Nf = Lf,Mf,Nf
        self.L,self.M,self.N = L,M,N
        self.T_db = T_db
        self.gamma_db = gamma_db
        self.average = average

        self.Ls,self.Ms,self.Ns = w_lambda.shape[0],w_mu.shape[0],w_nu.shape[0]
        self.L_nls,self.M_nls,self.N_nls = 8,8,8
        self.grid_lambda = arange(self.L)*2*pi/Lf
        self.grid_mu = arange(self.M)*2*pi/Mf
        self.grid_nu = arange(self.N)*2*pi/Nf
        self.grid_lambda_nls = linspace(0,L*2*pi/Lf,self.L_nls)
        self.grid_mu_nls = linspace(0,M*2*pi/Mf,self.M_nls)
        self.grid_nu_nls = linspace(0,self.N*2*pi/Nf,self.N_nls)

        self.W_lambda = ModelMatrixFourierDomain(self.w_lambda,self.grid_lambda)
        self.W_mu =     ModelMatrixFourierDomain(self.w_mu,self.grid_mu)           
        self.W_nu =     ModelMatrixFourierDomain(self.w_nu,self.grid_nu)            

        if (w_omega is not None) and (Of is not 1) and (O is not 1):
            self.w_omega,self.Of,self.O = w_omega,Of,O

            self.Os = w_omega.shape[0]
            self.os = arange(self.Os)
            self.op = (self.O-1)//2
            self.O_nls = 8

            self.grid_omega = arange(self.O)*2*pi/Of
            self.grid_omega_nls = linspace(0,self.O*2*pi/Of,self.O_nls)
            self.W_omega = ModelMatrixFourierDomain(self.w_omega,self.grid_omega)
            self.dfe = DecoupledFrequencyEstimation((self.L,self.M,self.N,self.O),
                                                    [
                                                     NonlinearLeastSquares(self.Ls,self.L,self.grid_lambda_nls,self.W_lambda),
                                                     NonlinearLeastSquares(self.Ms,self.M,self.grid_mu_nls,self.W_mu),
                                                     NonlinearLeastSquares(self.Ns,self.N,self.grid_nu_nls,self.W_nu),
                                                     NonlinearLeastSquares(self.Os,self.O,self.grid_omega_nls,self.W_omega)
                                                     ],
                                                    [
                                                     FrequencyEstimationPeakBased(self.w_lambda,self.Lf,self.grid_lambda),
                                                     FrequencyEstimationPeakBased(self.w_mu,self.Mf,self.grid_mu),
                                                     FrequencyEstimationPeakBased(self.w_nu,self.Nf,self.grid_nu),
                                                     FrequencyEstimationPeakBased(self.w_omega,self.Of,self.grid_omega)
                                                    ],
                                                    [self.W_lambda,self.W_mu,self.W_nu,self.W_omega])
            self.Md = 4
            self.psi_hat = zeros((4,2))
            self.peak_selector = tuple((array([self.L,self.M,self.N,self.O])-1)//2)

        else:

            self.dfe = DecoupledFrequencyEstimation((self.L,self.M,self.N),
                                                    [
                                                     NonlinearLeastSquares(self.Ls,self.L,self.grid_lambda_nls,self.W_lambda),
                                                     NonlinearLeastSquares(self.Ms,self.M,self.grid_mu_nls,self.W_mu),
                                                     NonlinearLeastSquares(self.Ns,self.N,self.grid_nu_nls,self.W_nu)],
                                                    [
                                                     FrequencyEstimationPeakBased(self.w_lambda,self.Lf,self.grid_lambda),
                                                     FrequencyEstimationPeakBased(self.w_mu,self.Mf,self.grid_mu),
                                                     FrequencyEstimationPeakBased(self.w_nu,self.Nf,self.grid_nu)
                                                    ],
                                                    [self.W_lambda,self.W_mu,self.W_nu])
            self.Md = 3
            self.psi_hat = zeros((3,2))
            self.peak_selector = tuple((array([self.L,self.M,self.N])-1)//2)
       
    def fit_single_target_model(self,X,dim):
        """
        Fits a single-target model in the peak neighborhood for the specified dimension. 

        X       -- 3D or 4D peak neighborhood in the Fourier domain, with 
                   dimensions corresponding to range, velocity, and 
                   angle (azimuth and/or elevation)
        dim     -- Dimension in which the model is fitted.

        mse     -- Mean squared error of the model fit.
        """

        s=X.shape
        sel_p = tuple((asarray(s)-1)//2)
        sl = slice_1d(self.peak_selector,dim)
        P = 20*log10(abs(X))
        
        R = self.dfe.covariance_matrix(X,dim)

        fe_peak_based = self.dfe.frequency_estimation_peak_based[dim]
        W =  self.dfe.model_matrices[dim]
        psi_hat_0 = fe_peak_based(self.peak_selector[dim],P[sl])
        self.psi_hat[dim,:] = psi_hat_0
        w_vec_psi = W(psi_hat_0)
        P_W  = dot(w_vec_psi,w_vec_psi.conj().T)/dot(w_vec_psi.conj().T,w_vec_psi)
        mse = real(trace(dot((eye(s[dim]) - P_W),R)))/s[dim]
        
        return mse

    def __call__(self,X,psi_a):
        """
        Processing steps as shown in Figure 2 in [1] extended to process a
        fourth frequency dimension corresponding to elevation angle.

        X       -- 3D or 4D peak neighborhood in the Fourier domain, with 
                   dimensions corresponding to range, velocity, and 
                   angle (azimuth and/or elevation)
        psi_a   -- lower border of the 3D or 4D frequency grid in the peak
                   neighborhood, with dimensions corresponding to range, velocity, and
                   angle (azimuth and/or elevation)

        psi_hat -- 3D or 4D frequency estimates, with dimensions corresponding to 
                   range, velocity, and angle (azimuth and/or elevation)

        Note, that two frequencies are returned even if the single-target model
        is accepted. In that case both estimates are identical.
        """
        self.mse_2,self.mse_1,self.psi_a = 100.,zeros(0),asarray(psi_a)
        self.dfe.Ta = 20*log10(abs(X[self.peak_selector])) - 6

        for dim in arange(self.Md):
            self.mse_1 = append(self.mse_1,self.fit_single_target_model(X,dim))
        
        res_dim = self.mse_1.argmax()  
        
        if 10*log10(self.mse_1[res_dim]) > self.T_db:
            self.mse_2 = self.dfe.fit_k_target_model(X,res_dim)

            if 10*log10(self.mse_1[res_dim]/self.mse_2) > self.gamma_db:
                self.psi_hat = self.dfe.remaining_dimensions()

        for dim in arange(self.Md):
            self.psi_hat[dim,:] += self.psi_a[dim]

        return self.psi_hat

class ModelMatrix:
    """
    Model matrix in the original domain

    Instance Variables

    Ns -- sample support

    """
    def __init__(self,Ns):
        self.Ns = Ns   
        self.ns = arange(self.Ns)

    def __call__(self,psi_k,A=None):
        if A is None:
            A = zeros((self.Ns,psi_k.shape[0]),complex_)
        exp(1j*outer(self.ns,psi_k),A)
        return A/sqrt(self.Ns)

    def derivative(self,psi_k,Ad=None):
        if Ad is None:
            Ad = zeros((psi_k.shape[0],self.Ns,psi_k.shape[0]),complex_)
        else:
            Ad *= 0
        for k,psi_k_c in enumerate(psi_k):
            Ad[k,:,k] = 1j*self.ns*exp(1j*self.ns*psi_k_c)/sqrt(self.Ns)
        return Ad

class ModelMatrixFourierDomain(ModelMatrix):
    """
    Model matrix in the Fourier domain

    Instance Variables

    w    -- window function
    N    -- support of Fourier transform
    grid -- frequency grid
    WF_H -- conjugate transpose of Fourier transform matrix

    """

    def __init__(self,w,grid):
        ModelMatrix.__init__(self,w.shape[0])
        self.w = w/sum(w)
        self.N = grid.shape[0]
        self.grid = grid
        self.WF_H = dot(diag(self.w),exp(1j*outer(self.ns,self.grid))).conj().T

    def __call__(self,psi_k,W=None):
        if W is None:
            W = zeros((self.N,psi_k.shape[0]),complex_)
        dot(self.WF_H, exp(1j*outer(self.ns,psi_k)),W)
        return W

    def derivative(self,psi_k,Wd=None):
        if Wd is None:
            Wd = zeros((psi_k.shape[0],self.N,psi_k.shape[0]),complex_)
        else:
            Wd *= 0
        for k in arange(psi_k.shape[0]):
            Wd[k,:,k] = dot(self.WF_H, 1j*self.ns*exp(1j*self.ns*psi_k[k]))
        return Wd

class NonlinearLeastSquares:
    """
    Nonlinear least squares (NLS) method for frequency estimation 

    Instance Variables

    Ns   -- sample support 
    N    -- support of Fourier transform
    grid -- frequency grid for evaluating the NLS criterion function
    K    -- Number of targets 
    P    -- projection matrix calculated on the frequency grid

    """

    def __init__(self,Ns,N,grid,W_ci,K=2):
        self.Ns,self.N,self.grid,self.K,self.N_grid = Ns,N,grid,K,grid.shape[0]
        self.W,self.W_dot = W_ci,W_ci.derivative
        self.P = zeros((self.N_grid**self.K,self.N,self.N),complex_)

        for i0 in arange(self.N_grid**self.K):
            im = asarray(unravel_index(i0,(self.N_grid,)*self.K))
            W = self.W(self.grid[im])
            self.P[i0,:,:] = dot(W,linalg.pinv(W))
        self.c = zeros(self.N_grid**self.K)

    def __call__(self,R):
        """
        Perform NLS optimization by evaluating the criterion function on a
        coarse grid, followed by Gauss-Newton iterations.
       
        R      -- covariance matrix

        psi_hat -- estimated frequencies
        mse    -- mean squared error of the K-target model fit

        """
        psi_hat_coarse = self._evaluate_on_grid(R)
        psi_hat = self._gauss_newton(R,psi_hat_coarse)
        
        return psi_hat % (2*pi)

    def _evaluate_on_grid(self,R):
        """
        Perform NLS optimization by evaluating the criterion function on a
        coarse grid.

        R      -- covariance matrix

        psi_hat -- estimated frequencies

        """
        self.c = real(trace(dot(self.P,R),axis1=1,axis2=2))
        im = asarray(unravel_index(self.c.argmax(),(self.N_grid,)*self.K))
        return self.grid[im]

    def _gauss_newton(self,R,psi_hat_init,N_it=4):
        """
        Perform NLS optimization by Gauss-Newton iterations.
 
        R           -- covariance matrix
        psi_hat_init -- starting point for iterations
        N_it        -- maximal number of iterations (default four)

        psi_hat      -- estimated frequencies
       
        """
        psi_hat = psi_hat_init.copy()
        g,H = zeros(self.K),diag(ones(self.K)) 
        W = zeros((self.N,self.K),complex_)
        Wd = zeros((self.K,self.N,self.K),complex_)

        for i in arange(N_it):
            W,Wd = self.W(psi_hat,W),self.W_dot(psi_hat,Wd)
            Wp = linalg.pinv(W)
            self._gradient(W,Wp,Wd,R,g)
            self._hessian(W,Wp,Wd,R,H)
            Hig = dot(linalg.pinv(H),g)
            sl = self._step_length(Hig,psi_hat,R)    
            psi_hat -= sl*Hig

            if sqrt(inner(Hig,Hig)) < 1e-5*2*pi/self.Ns:
                break
        return psi_hat

    def _gradient(self,W,Wp,Wd,R,g):
        """
        Calculate the gradient of the NLS criterion function

        W  -- model matrix for current iteration 
        Wp -- pseudo inverse of model matrix for current iteration  
        Wd -- derivative of model matrix for current iteration  
        R  -- covariance matrix

        g  -- gradient
         
        """
        P = eye(self.N) - dot(W,Wp)
        PR = dot(P,R)
        for i in arange(self.K):
            g[i] = -2*real(trace(dot(Wp.conj().T,dot(Wd[i,:,:].conj().T,PR))))

    def _hessian(self,W,Wp,Wd,R,H):
        """
        Calculate approximate Hessian matrix of the NLS criterion function

        W  -- model matrix for current iteration 
        Wp -- pseudo inverse of model matrix for current iteration  
        Wd -- derivative of model matrix for current iteration  
        R  -- covariance matrix

        H  -- approximated Hessian matrix 
         
        """

        P = eye(self.N) - dot(W,Wp)
        p_hat = dot(Wp,dot(R,Wp.conj().T))
        for i in arange(self.K):
            for j in arange(self.K):
                H[i,j] = 2.0*real(trace(dot(Wd[i,:,:].conj().T,dot(P,dot(Wd[j,:,:],p_hat)))))

    def _step_length(self,Hig,psi_hat,R):
        """
        Determine step length for update in current iteration

        Hig    -- product of the inverse Hessian matrix and the gradient for current iteration
        psi_hat -- estimated frequencies for current iteration 
        R      -- covariance matrix
     
        """
        c_last,sl_last = 1000.,0.
        for sl in 0.5**arange(10):
            W = self.W(psi_hat - sl*Hig)
            P =  eye(self.N) - dot(W,linalg.pinv(W))
            c = real(trace(dot(P,R)))
            if c_last <= c:
                break
            else:
                c_last = c
                sl_last = sl
        return sl_last


class FrequencyEstimationPeakBased:
    """
    Frequency estimation via peak position refinement.

    Sub-grid accuracy is achieved by a look-up table (LUT) approach.

    Instance Variables

    w      -- window function
    Nf     -- support of Fourier transform
    grid   -- local frequency grid in the vicinity of the periodogram peak
    lut    -- LUT for sub-grid shift over logarithmic peak power ratio
    sz_lut -- step size of logarithmic power ratio values

    """
    def __init__(self,w,Nf,grid,N_lut=100):
        self.w = w
        self.Nf = Nf
        self.grid = grid
        self.lut = None
        self.sz_lut = None
        self.calc_lut(N_lut)

    def calc_lut(self,N_lut=100):
        """
        Calculate LUT for sub-grid shift over logarithmic peak power ratio
       
        N_lut -- number of entries in the LUT

        """
        N = self.w.shape[0]
        psi_s = 2*pi*linspace(0.5,0.0,N_lut)/self.Nf
        x = exp(1j*outer(psi_s,arange(N)))
        P = 20.*log10(abs(fft.fft(self.w*x,int(self.Nf),axis=1)))
        ratio = P[:,0] - P[:,1]
        if ratio[-1] == inf:
            ratio[-1] = -psi_s[-3]*(ratio[-2]-ratio[-3])/(psi_s[-2]-psi_s[-3]) + ratio[-3]
        sz_lut = (ratio.max()-ratio.min())/(N_lut-1)
        lut = zeros(N_lut+1)
        ratio /= sz_lut
        for i in arange(N_lut):
            i_min = abs(i-ratio).argmin()
            if i_min == N_lut-1:
                i_min = N_lut-2
            lut[i] = psi_s[i_min] + (psi_s[i_min+1]-psi_s[i_min])*(i-ratio[i_min])/(ratio[i_min+1]-ratio[i_min])
        self.lut = lut
        self.sz_lut = sz_lut

    def __call__(self,maximizer,P,grid=None):
        """
        Peak-based frequency estimation with sub-grid accuracy
       
        maximizer -- peak frequency on grid
        P         -- logarithmic periodogram, i.e power spectrum
        grid      -- local frequency grid in the vicinity of the periodogram peak

        psi_hat   -- frequency, i.e. sub-grid peak position
        """

        if grid is None:
            grid = self.grid
        N = P.shape[0]
        N_lut = self.lut.size -1
        ratio_l = array([P[maximizer] - P[(maximizer - 1) % N]])
        ratio_r = array([P[maximizer] - P[(maximizer + 1) % N]])
        ratio = minimum(ratio_l,ratio_r)
        i_lut = ratio/self.sz_lut
        i_lut[i_lut > N_lut-1] = N_lut-1
        i_lut_g = floor(i_lut).astype('int')
        psi_s_lut = self.lut[i_lut_g] + (self.lut[i_lut_g+1]-self.lut[i_lut_g])*(i_lut-i_lut_g)
        psi_hat = (grid[maximizer] + sign(ratio_l-ratio_r)*psi_s_lut) % (2*pi)
        return psi_hat 


class DecoupledFrequencyEstimation:
    """
    Decoupled frequency estimation for an arbitrary number of data dimensions. 

    s                                    -- Shape of the data
    frequency_estimation_high_resolution -- List of high-resolution estimators for 
                                            model fitting in the resolution dimension.  
    frequency_estimation_peak_based      -- List of peak-based estimators for model fitting
                                            in the remaining dimensions.
    model_matrices                       -- List of model matrices per dimension. 
    K                                    -- Model order, i.e. number of targets. Note 
                                            that the current implementation is restricted to K=2.
    Ta                                   -- Threshold for covariance matrix averaging over the 
                                            remaining dimensions .

    psi_hat                              -- Frequency estimates 
    """

    def __init__(self,s,
                 frequency_estimation_high_resolution,
                 frequency_estimation_peak_based,
                 model_matrices,K=2,
                 Ta=-100):
        self.frequency_estimation_high_resolution = frequency_estimation_high_resolution
        self.frequency_estimation_peak_based = frequency_estimation_peak_based
        self.model_matrices = model_matrices
        self.s,self.K = s,K
        self.Ta = Ta
        self.psi_hat = zeros((len(s),self.K))

    def _swap_shape(self,s,dim):
        s_swap = asarray(s)
        s_swap[0],s_swap[dim] = s_swap[dim],s_swap[0]
        s_swap = tuple(s_swap)
        return s_swap

    def _swap(self,X,s=None,dim=None):
        if (s is not None) and (dim is not None):
            s_swap = asarray(self._swap_shape(s,dim))
        else: 
            s_swap=asarray(self.s_swap)
            dim = self.res_dim
        s_flat = (s_swap[0],prod(s_swap[1:]))
        return reshape(swapaxes(X,0,dim),s_flat)

    def _set_resolution_dimension(self,res_dim):
        self.res_dim = res_dim
        self.s_swap = self._swap_shape(self.s,res_dim)
        self.idx_o = arange(len(self.s))
        self.idx_o[0],self.idx_o[self.res_dim] = self.res_dim,0
        self.W = self.model_matrices[self.res_dim]

    def covariance_matrix(self,X,dim=None,Ta=None):
        """
        Covariance matrix estimation in the peak neighborhood. 

        X        -- Multidimensional data
        dim      -- Data dimension; if not specified the first dimension is used
        Ta       -- Threshold for averaging in the remaining dimensions

        returns  -- Covariance matrix in given dimension
        """
        if Ta is None:
            Ta = self.Ta
        if dim is not None:
            X = self._swap(X,X.shape,dim)
        P = 20*log10(abs(X))
        sel = P[P.argmax(axis=0),arange(X.shape[1])] >= Ta
        Xs = X[:,sel]
        return dot(Xs,Xs.conj().T)/Xs.shape[1]
 

    def fit_k_target_model(self,X,res_dim):
        """
        Frequency estimation in the resolution dimension by fitting a K-target model. 

        The resulting frequencies are stored internally and are subsequently used to
        obtain corresponding frequencies in the remaining dimensions.

        X        -- Multidimensional data
        res_dim  -- Resolution dimension

        mse      -- Mean squared error of the model fit.
        """
        self._set_resolution_dimension(res_dim)
        self.X = self._swap(X)

        R = self.covariance_matrix(self.X)
        psi_hat = self.frequency_estimation_high_resolution[self.res_dim](R)
        Wk  = self.W(psi_hat)
        P   = eye(self.s_swap[0]) - dot(Wk,linalg.pinv(Wk))
        mse = real(trace(dot(P,R)))/self.s_swap[0]
        self.psi_hat[self.res_dim,:] = psi_hat

        return mse

    def remaining_dimensions(self,X=None,psi_res_dim=None):
        """
        Frequency estimation in remaining dimension based on resolved frequencies 
        in the resolution dimension.

        """
        if psi_res_dim is None:
            psi_res_dim = self.psi_hat[self.res_dim,:]
        if X is None:
            X=self.X

        P = 20*log10(abs(dot(linalg.pinv(self.W(psi_res_dim)),X)))
        P = reshape(P,(self.K,) + self.s_swap[1:])

        for k in arange(self.K):
            Pk = P[k,:] 
            im = unravel_index(Pk.argmax(),self.s_swap[1:])
            for d in arange(len(self.s)-1):
                sl = slice_1d(im,d)
                self.psi_hat[self.idx_o[d+1],k] = self.frequency_estimation_peak_based[self.idx_o[d+1]](im[d],Pk[sl])
            
        return self.psi_hat

def slice_1d(i,dim):
    sl = list(i)
    sl[dim] = slice(None)
    return sl


