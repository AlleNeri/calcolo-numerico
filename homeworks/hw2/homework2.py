import numpy
import scipy
import matplotlib.pyplot as plt
import time

def errInNorma2(xTrue,x):
    num=scipy.linalg.norm(x-xTrue)
    denum=scipy.linalg.norm(xTrue)
    return num/denum

def raggioSpettrale(A):
	autoval=numpy.linalg.eigvals(A)
	return max(numpy.abs(autoval))

def metodiIterativiCheck(A):
    D=numpy.diag(A)
    definiti=True
    if(0 in D):
        definiti=False
    print("I metodi iterativi di Jacobi e Gauss-Seidel"," "if definiti else" non ","sono definiti sulla matrice data in quanto D e D-E sono"," non "if definiti else" ","singolari.",sep="")
    D=numpy.diag(D)
    E=-numpy.tril(A,k=-1)
    F=-numpy.triu(A,k=1)
    matriceDiIterazioneJ=numpy.dot(numpy.linalg.inv(D),(E+F))
    ro=raggioSpettrale(matriceDiIterazioneJ)
    print("Il raggio spettrale della matrice di iterazione J del metodi di Jacobi è:",ro,".")
    print("Si ha quindi che il metodo di Jacobi ",end="")
    if ro < 1:
        print("converge.")
    else:
        print("non converge.")
    matriceDiIterazioneGS=numpy.dot(numpy.linalg.inv(D-E),F)
    ro=raggioSpettrale(matriceDiIterazioneGS)
    print("Il raggio spettrale della matrice di iterazione L1 del metodo di Gauss-Seidel è:",ro,".")
    print("Si ha quindi che il metodo di Gauss-Seidel ",end="")
    if ro < 1:
        print("converge.")
    else:
        print("non converge.")

def fattorizzazioneLU(A,b,xTrue):
    cond=numpy.linalg.cond(A)
    LU, P=scipy.linalg.lu_factor(A)
    x=scipy.linalg.lu_solve((LU,P), b)
    errNorma2=errInNorma2(xTrue,x)
    return (cond,x,errNorma2)

def fastLU(A,b):
    LU, P=scipy.linalg.lu_factor(A,overwrite_a=True)
    x=scipy.linalg.lu_solve((LU,P), b, overwrite_b=True)
    return x

def fattorizzazioneCholesky(A,b,xTrue):
    cond=numpy.linalg.cond(A)
    L=scipy.linalg.cholesky(A, lower=True)
    y=scipy.linalg.solve(L, b, lower=True)
    Lt=numpy.transpose(L)
    x=scipy.linalg.solve(Lt, y)
    errNorma2=errInNorma2(xTrue,x)
    return (cond,x,errNorma2)

def fastCholesky(A,b):
    L=scipy.linalg.cholesky(A, lower=True, overwrite_a=True)
    y=scipy.linalg.solve(L, b, lower=True, overwrite_b=True)
    Lt=numpy.transpose(L)
    x=scipy.linalg.solve(Lt, y, overwrite_a=True, overwrite_b=True)
    return x

def Jacobi(A,b,x0,maxit,tol, xTrue):
	n=numpy.size(x0)
	ite=0
	x = numpy.copy(x0)
	err_iter=1+tol
	relErr=numpy.zeros((maxit, 1))
	errIter=numpy.zeros((maxit, 1))
	relErr[0]=errInNorma2(xTrue,x0)
	while (ite<maxit-1 and err_iter>tol):
		x_old=numpy.copy(x)
		for i in range(0,n):
			x[i]=(b[i]-numpy.dot(A[i,0:i],x_old[0:i])-numpy.dot(A[i,i+1:n],x_old[i+1:n]))/A[i,i]
		err_iter=errInNorma2(x_old,x)
		relErr[ite]=errInNorma2(xTrue,x)
		errIter[ite]=err_iter
		ite=ite+1
	relErr=relErr[:ite]
	errIter=errIter[:ite]  
	return [x, ite, relErr, errIter]

def fastJacobi(A,b,x0,maxit,tol):
    n=numpy.size(x0)     
    ite=0
    x = numpy.copy(x0)
    err_iter=1+tol
    while (ite<maxit-1 and err_iter>tol):
        x_old=numpy.copy(x)
        for i in range(0,n):
            x[i]=(b[i]-numpy.dot(A[i,0:i],x_old[0:i])-numpy.dot(A[i,i+1:n],x_old[i+1:n]))/A[i,i]
        ite=ite+1
        err_iter = errInNorma2(x_old,x)
    return x

def GaussSidel(A,b,x0,maxit,tol, xTrue):
	n=numpy.size(x0)
	ite=0
	x = numpy.copy(x0)
	err_iter=1+tol
	relErr=numpy.zeros((maxit, 1))
	errIter=numpy.zeros((maxit, 1))
	relErr[0]=errInNorma2(xTrue,x0)
	while(ite<maxit-1 and err_iter>tol):
		x_old=numpy.copy(x)
		for i in range(0,n):
			x[i]=(b[i]-A[i,0:i]@x[0:i]-A[i,i+1:n]@x_old[i+1:n])/A[i,i]
		ite+=1
		err_iter=errInNorma2(x_old,x)
		relErr[ite]=errInNorma2(xTrue,x)
		errIter[ite-1]=err_iter
	relErr=relErr[:ite]
	errIter=errIter[:ite]
	return [x, ite, relErr, errIter]

def fastGS(A,b,x0,maxit,tol):
    n=numpy.size(x0)
    ite=0
    x = numpy.copy(x0)
    err_iter=1+tol
    while(ite<maxit-1 and err_iter>tol):
        x_old=numpy.copy(x)
        for i in range(0,n):
            x[i]=(b[i]-A[i,0:i]@x[0:i]-A[i,i+1:n]@x_old[i+1:n])/A[i,i]
        ite+=1
        err_iter=errInNorma2(x_old,x)
    return x



def esLU():
    DIM_MIN=10
    DIM_MAX=1000

    condArr=[]
    errArr=[]

    for DIM in range(DIM_MIN,DIM_MAX):
        A=numpy.random.randn(DIM,DIM)
        b=A@numpy.ones((DIM,1))

        (cond,x,err)=fattorizzazioneLU(A,b,numpy.ones((DIM,1)))
        
        condArr.append(cond)
        errArr.append(err)

    plt.figure(figsize=(10,5))

    figura1=plt.subplot(2,1,1)
    figura1.plot([i for i in range(DIM_MIN,DIM_MAX)],condArr)
    figura1.grid()
    plt.title('CONDIZIONAMENTO DI A')
    plt.ylabel('K(A)')

    figura2=plt.subplot(2,1,2)
    figura2.plot([i for i in range(DIM_MIN,DIM_MAX)],errArr)
    figura2.grid()
    plt.title('ERRORE DELLA SOLUZIONE CON LU')
    plt.xlabel('dimensione matrice: n')
    plt.ylabel('errore in norma 2')

    plt.show()

def esCholeskyHilbert():
    DIM_MIN=2
    DIM_MAX=14

    condArr=[]
    errArr=[]

    for DIM in range(DIM_MIN,DIM_MAX):
        A=scipy.linalg.hilbert(DIM)
        b=A@numpy.ones((DIM,1))

        (cond,x,err)=fattorizzazioneCholesky(A, b, numpy.ones((DIM,1)))

        condArr.append(cond)
        errArr.append(err)
        
    plt.figure(figsize=(10, 5)) 

    figura1=plt.subplot(1,2,1)
    figura1.plot([i for i in range(DIM_MIN,DIM_MAX)],condArr)
    figura1.grid()
    plt.title('CONDIZIONAMENTO DI A')
    plt.xlabel('dimensione matrice: n')
    plt.ylabel('K(A)')

    figura2=plt.subplot(1,2,2)
    figura2.plot([i for i in range(DIM_MIN,DIM_MAX)],errArr)
    figura2.grid()
    plt.title('ERRORE DELLA SOLUZIONE DI CHOLESKY')
    plt.xlabel('dimensione matrice: n')
    plt.ylabel('errore in norma 2')

    plt.show()

def esCholeskyTridiag():
    DIM_MIN=10
    DIM_MAX=200

    condArr=[]
    errArr=[]

    for DIM in range(DIM_MIN,DIM_MAX):
        A=numpy.diag([9 for i in range(DIM)])+numpy.diag([-4 for i in range(DIM-1)],1)+numpy.diag([-4 for i in range(DIM-1)],-1)
        b=A@numpy.ones((DIM,1))

        (cond,_,err)=fattorizzazioneCholesky(A, b, numpy.ones((DIM,1)))

        condArr.append(cond)
        errArr.append(err)

    plt.figure(figsize=(10, 5)) 

    figura1=plt.subplot(1,2,1)
    figura1.plot([i for i in range(DIM_MIN,DIM_MAX)],condArr)
    figura1.grid()
    plt.title('CONDIZIONAMENTO DI A')
    plt.xlabel('dimensione matrice: n')
    plt.ylabel('K(A)')

    figura2=plt.subplot(1,2,2)
    figura2.plot([i for i in range(DIM_MIN,DIM_MAX)],errArr)
    figura2.grid()
    plt.title('ERRORE DELLA SOLUZIONE DI CHOLESKY')
    plt.xlabel('dimensione matrice: n')
    plt.ylabel('errore in norma 2')

    plt.show()

def esJacGS_DimFissa():
    N=100

    A = numpy.diag([9 for i in range(N)])+numpy.diag([-4 for i in range(N-1)],1)+numpy.diag([-4 for i in range(N-1)],-1)

    metodiIterativiCheck(A)

    xTrue = numpy.ones((N,1))
    b = A@xTrue
    x0 = numpy.zeros((N,1))
    x0[1]=1
    maxit = 500
    tol = 1.e-8

    (xJacobi, nIterJacobi, relErrJacobi, errIterJacobi) = Jacobi(A,b,x0,maxit,tol,xTrue)
    (xGS, nIterGS, relErrGS, errIterGS) = GaussSidel(A,b,x0,maxit,tol,xTrue)

    arrIterJacobi=[i for i in range(nIterJacobi)]
    arrIterGS=[i for i in range(nIterGS)]

    plt.plot(arrIterJacobi,relErrJacobi,label="Jacobi")
    plt.plot(arrIterGS,relErrGS,label="Gauss-Seidel")
    plt.legend()
    plt.grid()
    plt.title("ERRORE RELATIVO DEI METODI DI JACOBI E GAUSS-SEIDEL")
    plt.xlabel('numero di iterazioni del metodo')
    plt.ylabel('errore in norma 2')

    plt.show()

def esJacGS_DimVariabile():
    DIM_MIN=10
    DIM_MAX=500
    r=range(DIM_MIN,DIM_MAX,10)

    errFinaleJacobi=[]
    errFinaleGS=[]

    arrNumIteJacobi=[]
    arrNumIteGS=[]

    for N in r:
        A = numpy.diag([9 for i in range(N)])+numpy.diag([-4 for i in range(N-1)],1)+numpy.diag([-4 for i in range(N-1)],-1)
        xTrue = numpy.ones((N,1))
        b = A@xTrue
        x0 = numpy.zeros((N,1))
        x0[1]=1
        maxit = 500
        tol = 1.e-8

        (xJacobi, nIterJacobi, relErrJacobi, errIterJacobi) = Jacobi(A,b,x0,maxit,tol,xTrue)
        (xGS, nIterGS, relErrGS, errIterGS) = GaussSidel(A,b,x0,maxit,tol,xTrue)

        errFinaleJacobi.append(relErrJacobi[-1])
        errFinaleGS.append(relErrGS[-1])

        arrNumIteJacobi.append(nIterJacobi)
        arrNumIteGS.append(nIterGS)

    DIM=[i for i in r]

    plt.figure(figsize=(20,10))

    fig1=plt.subplot(1,2,1)
    fig1.plot(DIM,errFinaleJacobi,label="Jacobi")
    fig1.plot(DIM,errFinaleGS,label="Gauss-Seidel")
    fig1.legend()
    fig1.grid()
    plt.title("ERRORE RELATIVO DEI METODI\nDI JACOBI E GAUSS-SEIDEL AL\nVARIARE DELLA DIMENSIONE")
    plt.xlabel('dimensione matrice: n')
    plt.ylabel('errore in norma 2')

    fig2=plt.subplot(1,2,2)
    fig2.plot(DIM,arrNumIteJacobi,label="Jacobi")
    fig2.plot(DIM,arrNumIteGS,label="Gauss-Seidel")
    fig2.legend()
    fig2.grid()
    plt.title("NUMERO DI ITERAZIONE DEI METODI\nDI JACOBI E GAUSS-SEIDEL AL\nVARIARE DELLA DIMENSIONE")
    plt.xlabel('dimensione matrice: n')
    plt.ylabel('errore in norma 2')

    plt.show()

def esTempi():
    DIM_MIN=50
    DIM_MAX=400
    
    r=range(DIM_MIN,DIM_MAX,5)

    timeLU=[]
    timeCholesky=[]
    timeJacobi=[]
    timeGS=[]

    for N in r:
        A=numpy.diag([9 for i in range(N)])+numpy.diag([-4 for i in range(N-1)],1)+numpy.diag([-4 for i in range(N-1)],-1)
        xTrue = numpy.ones((N,1))
        b = A@xTrue
        x0 = numpy.zeros((N,1))
        x0[1]=1
        maxit = 500
        tol = 1.e-8

        startTime=time.time_ns()
        _=fastLU(A, b)
        timeLU.append(time.time_ns()-startTime)

        startTime=time.time_ns()
        _=fastCholesky(A, b)
        timeCholesky.append(time.time_ns()-startTime)

        startTime=time.time_ns()
        _=fastJacobi(A, b, x0, maxit, tol)
        timeJacobi.append(time.time_ns()-startTime)

        startTime=time.time_ns()
        _=fastGS(A, b, x0, maxit, tol)
        timeGS.append(time.time_ns()-startTime)

    errLU=[]
    errCholesky=[]
    errJacobi=[]
    errGS=[]

    for N in r:
        A=numpy.diag([9 for i in range(N)])+numpy.diag([-4 for i in range(N-1)],1)+numpy.diag([-4 for i in range(N-1)],-1)
        xTrue = numpy.ones((N,1))
        b = A@xTrue
        x0 = numpy.zeros((N,1))
        x0[1]=1
        maxit = 500
        tol = 1.e-8

        (_,_,err)=fattorizzazioneLU(A,b,xTrue)
        errLU.append(err)

        (_,_,err)=fattorizzazioneCholesky(A, b, xTrue)
        errCholesky.append(err)

        (_,_,err,_)=Jacobi(A,b,x0,maxit,tol,xTrue)
        errJacobi.append(err[-1])

        (_,_,err,_)=GaussSidel(A,b,x0,maxit,tol,xTrue)
        errGS.append(err[-1])

    DIM=[i for i in r]

    plt.figure(figsize=(20,10))

    fig1=plt.subplot(2,1,1)

    fig1.plot(DIM,errLU,label="LU")
    fig1.plot(DIM,errCholesky,label="Cholesky")
    fig1.plot(DIM,errJacobi,label="Jacobi")
    fig1.plot(DIM,errGS,label="Gauss-Seidel")
    fig1.legend()
    fig1.grid()
    plt.title("ERRORE DEI METODI SULLA STESSA MATRICE TEST")
    plt.xlabel("Dimensione della matrice")
    plt.ylabel("Errore in norma 2")
    
    fig2=plt.subplot(2,1,2)

    fig2.plot(DIM,timeLU,label="LU")
    fig2.plot(DIM,timeCholesky,label="Cholesky")
    fig2.plot(DIM,timeJacobi,label="Jacobi")
    fig2.plot(DIM,timeGS,label="Gauss-Seidel")
    fig2.legend()
    fig2.grid()
    plt.title("TEMPO DI ESECUZIONE DEI METODI SULLA STESSA MATRICE TEST")
    plt.xlabel("Dimensione della matrice")
    plt.ylabel("Tempo di esecuzione delle fattorizzazioni (ns)")

    plt.show()


def main():
    output="Selezionare un numero.\n1: Fattorizzazione LU\n2: Fattorizzazione di Cholesky su matrice di Hilbert\n3: Fattorizzazione di Cholesky su matrice tridiagonale\n4: Metodi di Jacobi e Gauss-Seidel su una matrice\n5: Metodi di Jacobi e Gauss-Seidel su matrici di dimensioni variabili\n6: Confrontando dei metodi\nScelta: "
    scelta=int(input(output))
    
    if(scelta==1): esLU()
    elif(scelta==2): esCholeskyHilbert()
    elif(scelta==3): esCholeskyTridiag()
    elif(scelta==4): esJacGS_DimFissa()
    elif(scelta==5): esJacGS_DimVariabile()
    elif(scelta==6): esTempi()
    else: print("Scelta non valida")

main()
