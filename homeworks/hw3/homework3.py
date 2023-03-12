import numpy
import matplotlib.pyplot as plt
import scipy
import pandas
import math

def fattorizzazioneCholesky(A,b):
    L=scipy.linalg.cholesky(A, lower=True, overwrite_a=True)
    y=scipy.linalg.solve(L, b, lower=True, overwrite_b=True)
    Lt=numpy.transpose(L)
    x=scipy.linalg.solve(Lt, y, overwrite_a=True, overwrite_b=True)
    return x

def equazioniNormali(A,b):
    AT=numpy.transpose(A)
    ATA = AT@A
    ATb = AT@b
    alpha_normali=fattorizzazioneCholesky(ATA, ATb)
    return alpha_normali

def svd(A,b):
    (U,sigma,VT)=scipy.linalg.svd(A)
    alpha_svd = numpy.zeros(sigma.shape)
    n=numpy.shape(sigma)[0]
    for i in range(n):
    	alpha_svd = alpha_svd + ( U[:,i] @ b * VT[i,:] / sigma[i] )
    return alpha_svd

def p(alpha, x):
    A = powerMatr(alpha.size-1,x)
    return A@alpha

def powerMatr(n,x):
    A=numpy.zeros((x.size,n+1))
    for i in range(n+1):
        A[:,i]=numpy.power(x,i)
    return A

def polinomio2metodi(n,x,y):
    A=powerMatr(n,x)

    alpha_normali=equazioniNormali(A, y)
    alpha_svd=svd(A,y)

    return (alpha_normali,alpha_svd)

def alphaPerGradi(N,x,y):
    alpha_normali=[]
    alpha_svd=[]
    
    for n in N:
        (normali,svd)=polinomio2metodi(n,x,y)

        alpha_normali.append(normali)
        alpha_svd.append(svd)

    return (alpha_normali,alpha_svd)

def graficiPolinomi(N,x,y,alpha_normali,alpha_svd,x_plot=numpy.linspace(1,3,100)):
    y_normali = []
    y_svd = []

    for i in N:
        y_normali.append([])
        y_svd.append([])
        for j in x_plot:
            y_normali[i-1].append(p(alpha_normali[i-1],j))
            y_svd[i-1].append(p(alpha_svd[i-1],j))

    plt.figure(figsize=(20, 10))

    col=len(N)
    row=2

    for n in N:
        plt.subplot(row, col, n)
        plt.plot(x,y,"o")
        plt.plot(x_plot,y_normali[n-1],label="Grado "+str(n))
        plt.grid()
        plt.legend()
        if(n==math.ceil(col/2)):
            plt.title('Approssimazione tramite Eq. Normali')

    for n in N:
        plt.subplot(row, col, len(N)+n)
        plt.plot(x,y,"o")
        plt.plot(x_plot,y_svd[n-1],label="Grado "+str(n))
        plt.grid()
        plt.legend()
        if(n==math.ceil(col/2)):
            plt.title('Approssimazione tramite SVD')

    plt.show()

def compressioneImmagineSVD(img,p_max=50):
    A=plt.imread(img)
    if(len(A.shape)!=2):
        A=A[:,:,0]
    plt.imshow(A, cmap="gray")
    plt.show()

    (U, sigma, VT)=scipy.linalg.svd(A)

    A_p=numpy.zeros(A.shape)

    err_rel=numpy.zeros((p_max))
    compressione=numpy.zeros((p_max))

    for i in range(p_max):
        A_p=A_p+numpy.outer(U[:,i],VT[i,:])*sigma[i]
        err_rel[i]=numpy.linalg.norm(A_p-A)/numpy.linalg.norm(A)
        compressione[i]=min(A.shape)/(i+1)-1
    
    print("Errore relativo alla ricostruzione di A: ",err_rel[-1])
    print("Fattore di comressione: ",compressione[-1])

    plt.imshow(A_p,cmap="gray")
    plt.show()

    plt.figure(figsize=(20,10))

    fig1=plt.subplot(2, 2, 1)
    fig1.plot(err_rel, 'o-')
    plt.grid()
    plt.title('Errore relativo')

    fig2=plt.subplot(2, 2, 2)
    fig2.plot(compressione, 'o-')
    plt.grid()
    plt.title('Fattore di compressione')
    
    fig3=plt.subplot(2, 2, 3)
    fig3.imshow(A, cmap='gray')
    plt.title('True image')

    fig4=plt.subplot(2, 2, 4)
    fig4.imshow(A_p, cmap='gray')
    plt.title('Reconstructed image with p=' + str(p_max))
    
    plt.show()


def es1():
    N = [i+1 for i in range(7)]
    x = numpy.array([1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3])
    y = numpy.array([1.18, 1.26, 1.23, 1.37, 1.37, 1.45, 1.42, 1.46, 1.53, 1.59, 1.5])

    (alpha_normali,alpha_svd)=alphaPerGradi(N,x,y)

    graficiPolinomi(N,x,y,alpha_normali,alpha_svd)

def es2():
    N = [i+1 for i in range(7)]
    dati=numpy.array(pandas.read_csv("HeightVsWeight.csv"))
    age=dati[:,0]
    weight=dati[:,1]

    (alpha_normali,alpha_svd)=alphaPerGradi(N,age,weight)

    graficiPolinomi(N,age,weight,alpha_normali,alpha_svd,x_plot=numpy.linspace(min(age)-1,max(age)+1,100))

def es3():
    output="1: funzione 1\n2: funzione 2\n3: funzione 3\nScelta: "
    scelta=int(input(output))

    fun=lambda x:x
    dominio=[]

    if(scelta==1):
        fun=lambda x: x*numpy.exp(x)
        dominio=[-1,1]
    elif(scelta==2):
        fun=lambda x: 1/(1+25*(x**2))
        dominio=[-1,1]
    elif(scelta==3):
        fun=lambda x: numpy.sin(5*x)+3*x
        dominio=[1,5]
    else:
        print("Scelta non valida")
        return

    gradi=[1,3,5,7]

    fig=plt.figure(figsize=(20,10))

    tot=len(gradi)
    col=math.ceil(tot/2)
    row=math.ceil(tot/col)

    for index in range(len(gradi)):
        m=15
        x=numpy.linspace(dominio[0],dominio[1],m)

        A=powerMatr(gradi[index],x)

        y_true=fun(x)

        alpha=svd(A,y_true)

        y_pol=p(alpha,x)

        ax=fig.add_subplot(row,col,index+1)
        ax.plot(x,y_true,"o",label="punti noti")
        ax.plot(x,y_true,label="funzione")
        ax.plot(x,y_pol,label="polinomio")
        plt.legend()
        plt.grid()
        plt.title("Polinomio di approssimazione di grado "+str(gradi[index]))

        controllo_err=1

        val_controllo_pol=p(alpha,numpy.array([controllo_err]))
        val_controllo_true=fun(controllo_err)

        err_punto=numpy.linalg.norm(val_controllo_true-val_controllo_pol,2)

        y_pol=p(alpha,numpy.array(x))

        err_complessivo=numpy.linalg.norm(y_true-y_pol,2)

        print("Errore in norma 2 nel punto x="+str(controllo_err)+" e complessivo: "+str(err_punto)+"\t"+str(err_complessivo))

    plt.show()

def es4():
    print("\tLuna")
    compressioneImmagineSVD("./moon.jpg")

def es5():
    print("\tCervello")
    compressioneImmagineSVD("./brain.jpg")
    print("\tTerra")
    compressioneImmagineSVD("./hearth.png",p_max=100)


def main():
    output="Selezionare un numero.\n1: Regressione polinomiale ai minimi quadrati(dataset 1)\n2: Regressione polinomiale ai minimi quadrati(dataset 2)\n3: Funzioni\n4: Immagine 1\n5: Immagine 2 e 3\nScelta: "
    scelta=int(input(output))
    
    if(scelta==1): es1()
    elif(scelta==2): es2()
    elif(scelta==3): es3()
    elif(scelta==4): es4()
    elif(scelta==5): es5()
    else: print("Scelta non valida")

main()
