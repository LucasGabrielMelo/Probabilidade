import numpy as np
from scipy.stats import poisson, uniform
from scipy.integrate import quad

## Funções matemáticas

def fatorial(n):
    if int(n) != n:
        raise ValueError("n deve ser inteiro")
    f = 1
    for k in range(1,n+1,1):
        f *= k
    return f

def binomial(n,k):
    if k > n:
        raise ValueError("k deve ser menor ou igual a n")
    num = fatorial(n)
    den = fatorial(k)*fatorial(n-k)
    return num/den

def integrando_funcao_Gamma(x,z):
    exp = np.exp(-x)
    polinomio = np.power(x, z-1)
    return polinomio * exp

def funcao_Gamma(Z):
    if np.isscalar(Z):
        Z = np.array([Z])
    resultados = []
    for z in Z:
        def f(x):
            return integrando_funcao_Gamma(x,z)
        resultado, erro = quad(f,0,np.inf)
        resultados.append(resultado)
    resultados = np.array(resultados)
    if len(resultados) == 1:
        return resultados[0]
    return resultados

## Eventos aleatórios discretos

def Bernoulli(p):
    if (p < 0) or (p > 1):
        raise ValueError("p deve estar entre 0 e 1")
    
    return np.random.choice([0,1], size = 1, p = [1-p,p])[0]

def Bin(n,p):
    if n <= 0:
        raise ValueError("n deve ser maior que 0")
    elif int(n) != n:
        raise ValueError("n deve ser inteiro")
    
    sucessos = 0
    for k in range(0,n,1):
        b = Bernoulli(p)
        if b == 1:
            sucessos += 1
    return sucessos

def Geom(p):
    k = 1
    while Bernoulli(p) == 0:
        k += 1
    return k

def Poisson(lambda_):
    if lambda_ < 0:
        raise ValueError("\u03BB deve ser maior ou igual a 0")
    elif int(lambda_) != lambda_:
        raise ValueError("\u03BB deve ser inteiro")
    
    return np.random.poisson(lam = lambda_)

## Distribuições de probabilidades discretas

def prob_Bin(n,p,k):
    bin = binomial(n,k)
    p_sucessos = p**k
    p_fracassos = (1-p)**(n-k)
    return bin * p_sucessos * p_fracassos

def prob_Poisson(lambda_, k):
    if lambda_ < 0:
        raise ValueError("\u03BB deve ser maior ou igual a 0")
    elif int(lambda_) != lambda_:
        raise ValueError("\u03BB deve ser inteiro")
    '''
    P[X=k] = \frac{e^{-lambda} * lambda^k}{k!}
    '''
    return poisson.pmf(k, mu = lambda_) # Utilizando essa função da scipy por questões de memória

def prob_Geom(p, k):
    return p*(1-p)**(k-1)

## Distribuições de probabilidades contínuas

def prob_Gaussiana(t, mu, sigma2):
    sigma = np.sqrt(sigma2)
    expoente = -(t-mu)**2 / (2*sigma2)
    den = sigma * np.sqrt(2*np.pi)
    return np.exp(expoente)/den