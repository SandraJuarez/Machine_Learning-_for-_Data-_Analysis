import jax
import jax_metrics as jm
import jax.numpy as jnp
from jax import grad, jit, vmap
from functools import partial
from jax import random
import os
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import graficar
# Switch off the cache 
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

class Linear_Model():
    """
    Basic Linear Regression with Ridge Regression
    """
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.key = random.PRNGKey(0)
        self.cpus = jax.devices("cpu")

    @staticmethod
    @jit
    def linear_model(X: jnp, theta: jnp) -> jnp:
        """
        Classic Linear Model. Jit has been used to accelerate the loops after the first one
        for the Gradient Descent part
        args:
            X: Data array at the GPU or CPU
            theta: Parameter w for weights and b for bias
        returns:
            f(x): the escalar estimation on vector x or the array of estimations
        """
        w = theta[:-1]
        b = theta[-1]
        return jax.numpy.matmul(X, w) + b

    def generate_theta(self):
        """
        Use the random generator at Jax to generate a random generator to instanciate
        the augmented values
        """
        keys = random.split(self.key, 1)
        return jax.numpy.vstack([random.normal(keys[0], (self.dim,1)), jax.numpy.array(0)])
        
    @partial(jit, static_argnums=(0,))
    def LSE(self, theta: jnp, X: jnp, y: jnp)-> jnp:
        """
        LSE in matrix form. We also use Jit por froze info at self to follow 
        the idea of functional programming on Jit for no side effects
        args:
            X: Data array at the GPU or CPU
            theta: Parameter w for weights and b for bias
            y: array of labels
        returns:
            the Loss function LSE under data X, labels y and theta initial estimation
        """
        return (jax.numpy.transpose(y - self.linear_model(X, theta))@(y - self.linear_model(X, theta)))[0,0]

    @partial(jit, static_argnums=(0,))
    def update(self, theta: jnp, X: jnp, y: jnp, lr):
        """
        Update makes use of the autograd at Jax to calculate the gradient descent.
        args:
            X: Data array at the GPU or CPU
            theta: Parameter w for weights and b for bias
            y: array of labels
            lr: Learning rate for Gradient Descent
        returns:
            the step update w(n+1) = w(n)-δ(t)𝜵L(w(n))        
        """
        return theta - lr * jax.grad(self.LSE)(theta, X, y)

        
    @partial(jit, static_argnums=(0,))
    def estimate_grsl(self, X, theta):
        """
        Estimation for the Gradient Descent version
        args:
            X: Data array at the GPU or CPU
            theta: Parameter w for weights and b for bias
        return:
            Estimation of data X under linear model
        """
        w = theta[:-1]
        b = theta[-1]
        return X@w+b
    
    def get_pr(self,k_classes,samples,clases,y0,y_hat):
        FP=0
        FN=0
        TP=0
        recall_list=[]
        samples=100
        precision_list=[]
        for k in range(k_classes):
            for i in range (samples):
                if y0[i]==clases[k] and y_hat[i]==clases[k]:
                    TP+=1
                if y0[i]!=clases[k] and y_hat[i] == clases[k]:
                    FP+=1
                if y0[i]==clases[k] and y_hat[i] != clases[k]:
                    FN+=1
            if FP+TP!=0:
                precision_list.append(TP/(TP+FP))
            else:
                precision_list.append(0)
            if FN+TP!=0:
                recall_list.append(TP/(TP+FN))
            else:
                recall_list.append(0)

            
            
    
        precision=sum(precision_list)/k_classes
        recall=sum(recall_list)/k_classes
        return precision,recall
    
    
    def gradient_descent(self, theta,  X, y, n_steps, lr = 0.001):
        """
        Gradient Descent Loop for the LSE Linear Model
        args:
            X: Data array at the GPU or CPU
            theta: Parameter w for weights and b for bias
            y: array of labels
            n_steps: number steps for the Gradient Loop
            lr: Learning rate for Gradient Descent   
        return:
            Updated Theta
        """
        for i in range(n_steps):
            theta = self.update(theta, X, y, lr)
        return theta
    @partial(jit, static_argnums=(0,))
    def model(self, theta, X, y, lr,n_steps,k_classes,samples,clases,X_val,y_val,run_name):
        experiment_name = "linear"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        with mlflow.start_run(experiment_id=experiment_id,run_name=run_name):
            mlflow.log_param('learning_rate',lr)
            mlflow.log_param('n_steps',n_steps)
            theta=self.gradient_descent(theta,X,y,n_steps,lr)
            y_hat=self.estimate_grsl(X, theta)
            y_hat_val=self.estimate_grsl(X_val,theta)
            precision,recall=self.get_pr(k_classes,samples,clases,y,y_hat)
            precision_val,recall_val=self.get_pr(k_classes,samples,clases,y_val,y_hat_val)
            mlflow.log_metric('precision',precision)
            mlflow.log_metric('recall',recall)
            mlflow.log_metric('precision_val',precision_val)
            mlflow.log_metric('recall_val',recall_val)
            print('In the training, the precision and recall are:',precision,recall)
            print('In the validation, the precision and recall are:',precision_val,recall_val)
        return y_hat,y_hat_val,precision,recall,precision_val,recall_val

        
    ######################################################################################################
    #########vamos a hacer la implementación de la regularización de Ridge##################################

    def generate_canonicalRidge_estimator(self, X: jnp, y:jnp,la:jnp) -> jnp:
        """
        Cannonical LSE error solution for the Linearly separable classes 
        args:
            X: Data array at the GPU or CPU
            y: Label array at the GPU 
        returns:
            w: Weight array at the GPU or CPU
        """
        XX=jax.numpy.transpose(X)@X
        dimension=int(jnp.shape(XX)[0])
        I=jax.numpy.identity(dimension)
        return  jax.numpy.linalg.inv(XX+la*I)@jax.numpy.transpose(X)@y
    
    @staticmethod
    def estimate_cannonicalRidge(X: jnp, w: jnp)->jnp:
        """
        Estimation for the Gradient Descent version
        args:
            X: Data array at the GPU or CPU
            w: Parameter w under extended space
        return:
            Estimation of data X under cannonical solution
        """
        return X@w
    
    def model_ridge(self,X,y_est,l,incremento,samples,k_classes,clases,X_v,y_est_v,run_name):
        experiment_name = "Ridge"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        X=X[samples,:]
        y_est=y_est[samples,:]
        X_e = np.hstack([X, np.ones((samples,1))])
        X_v= np.hstack([X_v, np.ones((samples,1))])
        l=0
        max_steps=25
        precision_list=[]
        recall_list=[]
        precision_list_v=[]
        recall_list_v=[]
        lbda=[]
        with mlflow.start_run(experiment_id= experiment_id, run_name=run_name) as run:
            mlflow.log_parameter('lambda',l)


            for i in range(max_steps):
                l=l+incremento
                lbda.append(l)
                wR = self.generate_canonicalRidge_estimator(X_e, y_est,l)
                y_hatR = self.estimate_cannonicalRidge(X_e, wR)
                y_hatR_v = self.estimate_cannonicalRidge(X_e, wR)
                precision,recall=self.get_pr(self,k_classes,samples,clases,y_est,y_hatR)
                precision_v,recall_v=self.get_pr(self,k_classes,samples,clases,y_est_v,y_hatR_v)
                precision_list.append(precision)
                recall_list.append(recall)
                precision_list_v.append(precision_v)
                recall_list_v.append(recall_v)
            mlflow.log_metric('precision_list',precision_list)
            mlflow.log_metric('recall_list',recall_list)
            mlflow.log_metric('precision_val',precision_list_v)
            mlflow.log_metric('recall_val',recall_list_v)
            grafica_pr=graficar.graficar_pr_ridge(recall_list,precision_list,recall_list_v,precision_list_v)
            mlflow.log_figure(grafica_pr,'grafica_pr_ridge.png')
        return y_hatR,y_hatR_v,precision_list