import matplotlib.pyplot as plt
import numpy as np
from statistics import stdev
e1=[]
# função do circuito rl => I(t)=Io*e^-((r/l)*t), r=1, l=1, e Io=1
def dydx(x): # 1ª derivada da função
    return -1*(1*(1/2))*np.exp(-1*(1/2)*x)
def dydx2(x): # 2ª derivada da função
    return (1*(1/4))*np.exp(-1*(1/2)*x)
def f(x):
    return 1*np.exp(-1*(1/2)*x) # função original
def erro(y,y1):
    erro=0
    for i in range(len(y)):
        e=((y[i]-y1[i])/y[i])**2
        erro+=e
        e1.append(e)
    return erro

def rungeKutta1(x0, y0, x, h):
    n = int((x - x0)/h)
    X=np.zeros(n+1)
    Y=np.zeros(n+1)
    Y[0] = y0
    X[0] = x0
    for i in range(0, n):
     k1 = h * dydx(X[i])
     #k2 = h * dydx(X[i] + 0.5 * h, Y[i] + 0.5 * k1) 
     #k3 = h * dydx(X[i] + 0.5 * h, Y[i] + 0.5 * k2) 
     #k4 = h * dydx(X[i] + h, Y[i] + k3) 
     #Y[i] = y[i] +(1.0/6)*(k1+2*k2+2*k3+k4)
     Y[i+1] = Y[i] +k1
     X[i+1] = X[i] + h
    Y1=f(X)
    soma=erro(Y1,Y)
    print("Erro=",soma," com h=0.001 soma dos erro=",soma/len(e1),"Desvio padrão=",stdev(e1))#Erro quadrático médio
    plt.plot(X,Y,'b-',X,Y1,'r-',X,e1,'g-')
    plt.legend(['Runge-Kutka 1ª ordem','Função Original','Função erro'])
    plt.axis([0,2,-0.2,2])
    plt.grid(True)
    plt.title("Solução da $I'=-1*(1*(1/2))*np.exp(-1*(1/2)*t) , I(0)=1$ r=1, l=2")
    plt.show()

def rungeKutta2(x0, y0, x, h):
    n = int((x - x0)/h)
    X=np.zeros(n+1)
    Y=np.zeros(n+1)
    Y[0] = y0
    X[0] = x0
    for i in range(0, n):
     k1 = h * dydx2(X[i])
     Y[i+1] = Y[i] +k1
     X[i+1] = X[i] + h
    Y1=dydx(X)
    soma=erro(Y1,Y)
    print("Erro=",soma," com h=0.001 soma dos erro=",soma/len(e1),"Desvio padrão=",stdev(e1))#Erro quadrático médio
    plt.plot(X,Y,'b-',X,Y1,'r-',X,e1,'g-')
    plt.legend(['Runge-Kutka 2ª ordem','Função da 1ª derivada(como a função exata)','Função erro'])
    plt.axis([0,2,-2,2])
    plt.grid(True)
    plt.title("Solução da $I''=(1*(1/4))*np.exp(-1*(1/2)*t) , I(0)=1$ r=1, l=2")
    plt.show()
 

x0 = 0
y = 1
x = 2
h = (2/21)
#rungeKutta1(x0, y, x, h)
rungeKutta2(x0,dydx(x0), x, h)

# This code is contributed by Prateek Bhindwar 


    
