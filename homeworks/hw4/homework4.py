import numpy as np
import math
import matplotlib.pyplot as plt

def segniOpposti(f,a,b):
    if(np.sign(f(a))*np.sign(f(b))<0):
        return True
    return False

def bisezione(a, b, f, tolx, xTrue):
    k = math.ceil(math.log2(abs(b-a)/tolx))
    errAss=[]
    if(not segniOpposti(f,a,b)):
        print("Non si può calcolare il metodo di Bisezione")
        return (None, None, k, errAss)
    for i in range(k):
        p_medio=(a+b)/2
        errAss.append(abs(p_medio-xTrue))
        if abs(f(p_medio))<=tolx:
            x=p_medio
            return (x, i, k, errAss)
        if np.sign(f(a))*np.sign(f(p_medio))<0:
            b=p_medio
        else:
            a=p_medio
    return (x, i, k, errAss)

def succ_app(f, g, tolf, tolx, maxit, xTrue, x0=0):
    errRel=[]
    errAss=[]

    i=0
    errRel.append(tolx+1)
    errAss.append(abs(x0-xTrue))
    x=x0

    while i < maxit and (errRel[i] > tolx or abs(f(x)) > tolf):
        x_new=g(x)
        errRel.append(abs(x_new-x))
        errAss.append(abs(x_new-xTrue))
        i=i+1
        x=xTrue

    return (x, i, errRel, errAss)

def newton(f, df, tolf, tolx, maxit, xTrue, x0=0):
    g=lambda x: x-(f(x)/df(x))

    (x,i,errRel,errAss)=succ_app(f, g, tolf, tolx, maxit, xTrue, x0)

    return (x,i,errRel,errAss)

def esZeriFunzioni(f,df,xTrue,a,b,gArr,tolx=10**(-10),tolf=10**(-6),maxit=100,x0=0):
    fTrue = f(xTrue)
    print("Errore di xTrue: ",fTrue,"\n")

    x_as=np.empty(len(gArr))
    i_as=np.empty(len(gArr))
    errRel_as=np.empty(len(gArr),dtype=object)
    errAss_as=np.empty(len(gArr),dtype=object)

    (x_b,i_b,k_b,errAss_b)=bisezione(a, b, f, tolx, xTrue)
    (x_n,i_n,errRel_n,errAss_n)=newton(f,df,tolf,tolx,maxit,xTrue,x0)
    for i in range(len(gArr)):
        (x_as[i],i_as[i],errRel_as[i],errAss_as[i])=succ_app(f,gArr[i],tolf,tolx,maxit,xTrue,x0)

    print('metodo di bisezione\nx=',x_b,'\niterazioni=',i_b,'\nerrore=',errAss_b[-1],'\n')
    print('metodo di newton\nx=',x_n,'\niterazioni=',i_n,'\nerrore=',errAss_n[-1],'\n')
    for i in range(len(gArr)):
        print('Metodo approssimazioni successive g'+str(i+1)+'\nx=',x_as[i],'\niterazioni=',i_as[i],"\nerrore assoluto=",errAss_as[i][-1],"\n")

    x_plot=np.linspace(a, b, 101)
    f_plot=[f(i) for i in x_plot]

    plt.figure(figsize=(20,10))

    fig1=plt.subplot(1,2,1)
    fig1.plot(x_b,0,"o",label="Bisezione")
    fig1.plot(x_n,0,"o",label="Newton")
    for i in range(len(gArr)):
        fig1.plot(x_as[i],0,"o",label="Approssimazioni successive "+str(i+1))
    fig1.plot(x_plot,f_plot)
    plt.legend()
    plt.grid()
    plt.title("Funzione")

    fig2=plt.subplot(1,2,2)
    fig2.plot([i for i in range(len(errAss_b))],errAss_b,"o-",label="Bisezione")
    fig2.plot([i for i in range(len(errAss_n))],errAss_n,"o-",label="Newton")
    for i in range(len(gArr)):
        fig2.plot([i for i in range(len(errAss_as[i]))],errAss_as[i],"o-",label="Approssimazioni successive "+str(i+1))
    plt.legend()
    plt.grid()
    plt.xlabel("Numero di iterazioni")
    plt.ylabel("Errore")
    plt.title("Errori")

    plt.show()

def next_step(x,f,grad,alphaMin=1e-7,jMax=10):
    alpha=1.1
    rho=0.5
    c1=0.25
    p=-np.array(grad)
    j=0
    while f((x+(alpha*p))) > f(x)+(c1*alpha*(np.transpose(grad)@p)) and alpha > alphaMin and j < jMax:
        alpha=alpha*rho
        j=j+1
    if j>=jMax or alpha < alphaMin:
        return -1
    else:
        return alpha

def minimize(x0,xTrue,f,grad_f,passoFisso=False,maxIteration=1000,absoluteStop=1e-5,max_it_next_step=10):
    x=[]
    normGradList=[]
    functionEvalList=[]
    errorList=[]

    k=0
    xLast=np.array([x0[0],x0[1]])
    x.append(xLast)
    functionEvalList.append(abs(f(xLast)))
    errorList.append(np.linalg.norm(np.array(xLast)-np.array(xTrue)))
    normGradList.append(np.linalg.norm(grad_f(xLast)))

    while (np.linalg.norm(grad_f(xLast))>absoluteStop and k < maxIteration):
        k=k+1

        grad=grad_f(xLast)
        step=0.1
        if(not passoFisso):
            step=next_step(xLast,f,grad,jMax=max_it_next_step)

        if(step==-1):
            print("Non converge!")
            return (xLast, normGradList, functionEvalList, errorList, k, x)

        xLast=xLast-(step*np.array(grad))

        x.append(xLast)
        functionEvalList.append(abs(f(xLast)))
        errorList.append(np.linalg.norm(xLast-xTrue))
        normGradList.append(np.linalg.norm(grad_f(xLast)))

    normGradList=np.array(normGradList)
    functionEvalList=np.array(functionEvalList)
    errorList=np.array(errorList)
    x=np.array(x)

    return (xLast, normGradList, functionEvalList, errorList, k, x)

def es1():
    f=lambda x: math.exp(x)-(x**2)
    df=lambda x: math.exp(x)-(2*x)
    xTrue=-0.7034674

    a=-1.
    b=1.

    g1 = lambda x: x-f(x)*np.exp(x/2)
    g2 = lambda x: x-f(x)*np.exp(-x/2)
    g3 = lambda x: x-f(x)/df(x)

    esZeriFunzioni(f,df,xTrue,a,b,[g1,g2,g3])

def es2():
    f=lambda x: (x**3)+(4*math.cos(x)*x)-2
    df=lambda x: (3*(x**2))+(4*math.cos(x))-(4*x*math.sin(x))
    xTrue = 0.5368385545949

    a=0.
    b=2.

    g1=lambda x: (2-(x**3))/(4*math.cos(x))

    esZeriFunzioni(f,df,xTrue,a,b,[g1])

def es3():
    f=lambda x: x-(x**(1/3))-2
    df=lambda x: 1-(1/(3*x**(2/3)))
    xTrue=3.5213797068269

    a=3.
    b=5.

    g1=lambda x: (x**(1/3))+2

    esZeriFunzioni(f,df,xTrue,a,b,[g1],x0=1)

def es4():
    f=lambda x: 10*((x[0]-1)**2)+((x[1]-2)**2)
    nabla_f=lambda x: [20*(x[0]-1),2*(x[1]-2)]
    x0=[3,-5]
    xTrue=[1,2]

    (xLast,normGradList,functionEvalList,errorList,k,x_y)=minimize(x0,xTrue,f,nabla_f)
    print("Risultato:",x_y[-1],"\n")

    x_plot=np.linspace(min(x_y[:,0])-1,max(x_y[:,0])+1)
    y_plot=np.linspace(min(x_y[:,1])-1,max(x_y[:,1])+1)
    X, Y=np.meshgrid(x_plot,y_plot)
    Z=f([X,Y])

    fig,ax=plt.subplots(subplot_kw={"projection":"3d"})
    surf=ax.plot_surface(X, Y, Z, cmap="viridis")
    ax.set_title('Superfice')
    fig.colorbar(surf)
    
    plt.show()

    plt.figure(figsize=(16, 10))

    plt.contour(X, Y, Z, levels=100)
    plt.plot(x_y[:,0],x_y[:,1])
    plt.title('Curve di livello')
    
    plt.show()

    iterPlot=[i for i in range(k+1)]

    plt.figure(figsize=(16,10))

    plt.subplot(3,1,1)
    plt.plot(iterPlot,normGradList,"o-")
    plt.grid()
    plt.ylabel('norma del gradiente')
    plt.title('Norma del gradiente al variare del numero di iterazioni')

    plt.subplot(3,1,2)
    plt.plot(iterPlot,errorList,"o-")
    plt.grid()
    plt.ylabel('errore')
    plt.title('Errore al variare delle iterazioni')

    plt.subplot(3,1,3)
    plt.plot(iterPlot,functionEvalList,"o-")
    plt.grid()
    plt.xlabel('iterazioni')
    plt.ylabel('valore funzione')
    plt.title('Valore della funzione obiettivo al variare delle iterazioni')

    plt.show()

def es5():
    x0=[10,100]

    l=0.3
    f=lambda x: (np.linalg.norm(x-np.ones(x.shape),2)**2)+(l*(np.linalg.norm(x,2)**2))
    nabla_f=lambda x: [ (2*(x[0]-1))+(l*2*x[0]), (2*(x[1]-1))+(l*2*x[1]) ]
    xTrue=[10/13,10/13]
    print("Esercizio con λ =",l)
    (xLast,normGradList,functionEvalList,errorList,k,x_y)=minimize(x0,xTrue,f,nabla_f)
    print("Risultato:",x_y[-1],"\n")

    iterPlot=[i for i in range(k+1)]

    plt.figure(figsize=(16,10))

    plt.subplot(2,1,1)
    plt.plot(iterPlot,normGradList,"o-")
    plt.grid()
    plt.ylabel('norma del gradiente')
    plt.title('Norma del gradiente al variare del numero di iterazioni')

    plt.subplot(2,1,2)
    plt.plot(iterPlot,functionEvalList,"o-")
    plt.grid()
    plt.xlabel('iterazioni')
    plt.ylabel('valore funzione')
    plt.title('Valore della funzione obiettivo al variare delle iterazioni')

    plt.show()

    l=0.5
    xTrue=[10/15,10/15]
    print("Esercizio con λ =",l)
    (xLast,normGradList,functionEvalList,errorList,k,x_y)=minimize(x0,xTrue,f,nabla_f)
    print("Risultato:",x_y[-1],"\n")

    iterPlot=[i for i in range(k+1)]

    plt.figure(figsize=(16,10))

    plt.subplot(2,1,1)
    plt.plot(iterPlot,normGradList,"o-")
    plt.grid()
    plt.ylabel('norma del gradiente')
    plt.title('Norma del gradiente al variare del numero di iterazioni')

    plt.subplot(2,1,2)
    plt.plot(iterPlot,functionEvalList,"o-")
    plt.grid()
    plt.xlabel('iterazioni')
    plt.ylabel('valore funzione')
    plt.title('Valore della funzione obiettivo al variare delle iterazioni')

    plt.show()

    l=0.7
    xTrue=[10/17,10/17]
    print("Esercizio con λ =",l)
    (xLast,normGradList,functionEvalList,errorList,k,x_y)=minimize(x0,xTrue,f,nabla_f)
    print("Risultato:",x_y[-1],"\n")

    iterPlot=[i for i in range(k+1)]

    plt.figure(figsize=(16,10))

    plt.subplot(2,1,1)
    plt.plot(iterPlot,normGradList,"o-")
    plt.grid()
    plt.ylabel('norma del gradiente')
    plt.title('Norma del gradiente al variare del numero di iterazioni')

    plt.subplot(2,1,2)
    plt.plot(iterPlot,functionEvalList,"o-")
    plt.grid()
    plt.xlabel('iterazioni')
    plt.ylabel('valore funzione')
    plt.title('Valore della funzione obiettivo al variare delle iterazioni')

    plt.show()


def main():
    output="Selezionare un numero.\n1: zero della funzione 1\n2: zero della funzione 2\n3: zero della funzione 3\n4: minimo della funzione 1\n5: minimo della funzione 2\nScelta: "
    scelta=int(input(output))
    
    if(scelta==1): es1()
    elif(scelta==2): es2()
    elif(scelta==3): es3()
    elif(scelta==4): es4()
    elif(scelta==5): es5()
    else: print("Scelta non valida")

main()
