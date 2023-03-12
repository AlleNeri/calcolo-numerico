import numpy as np
import matplotlib.pyplot as plt
import scipy
from skimage import data, metrics

np.random.seed(0)

# Crea un kernel Gaussiano di dimensione kernelLen e deviazione standard sigma
def gaussian_kernel(kernelLen, sigma):
    x = np.linspace(- (kernelLen // 2), kernelLen // 2, kernelLen)    
    # Kernel gaussiano unidmensionale
    kern1d = np.exp(- 0.5 * (x**2 / sigma))
    # Kernel gaussiano bidimensionale
    kern2d = np.outer(kern1d, kern1d)
    # Normalizzazione
    return kern2d / kern2d.sum()

# Esegui l'fft del kernel K di dimensione d agggiungendo gli zeri necessari 
# ad arrivare a dimensione shape
def psf_fft(Kernel, deviazioneStandard, shape):
    # Aggiungi zeri
    K_p = np.zeros(shape)
    K_p[:deviazioneStandard, :deviazioneStandard] = Kernel

    # Sposta elementi
    p = deviazioneStandard // 2
    K_pr = np.roll(np.roll(K_p, -p, 0), -p, 1)

    # Esegui FFT
    K_otf = np.fft.fft2(K_pr)
    return K_otf

# Moltiplicazione per A
def A(x, K):
  x = np.fft.fft2(x)
  return np.real(np.fft.ifft2(K * x))

# Moltiplicazione per A trasposta
def AT(x, K):
  x = np.fft.fft2(x)
  return np.real(np.fft.ifft2(np.conj(K) * x))

def problemaTest(l=0):
    imgOriginale = data.camera().astype(np.float64)/255.0
    m,n=imgOriginale.shape

    #generazione filtro di blur per test
    filtroBlur=psf_fft(gaussian_kernel(24, 3),24,imgOriginale.shape)

    #generazione rumore per test
    sigma = 0.02
    noise = np.random.normal(0,sigma,size=imgOriginale.shape)

    #blur e noise, immagine corrotta
    imgCorrotta = A(imgOriginale,filtroBlur)+noise

    PSNR=metrics.peak_signal_noise_ratio(imgOriginale, imgCorrotta)
    MSE=metrics.mean_squared_error(imgOriginale,imgCorrotta)

    f=lambda x: 0.5*((np.sum(np.square((A(np.reshape(x,(m,n)),filtroBlur)-imgCorrotta))))+(l*(np.linalg.norm(x,2)**2)))

    df=lambda x: x
    if(l==0):
        df=lambda x: np.reshape(AT(A(np.reshape(x,(m,n)),filtroBlur),filtroBlur)-AT(imgCorrotta,filtroBlur),m*n)
    else:
        df=lambda x: np.reshape(AT(A(np.reshape(x,(m,n)),filtroBlur),filtroBlur)-AT(imgCorrotta,filtroBlur)+np.reshape(x, (m, n)),m*n)

    x0=imgCorrotta
    max_it=25

    imgRicostruita_arr=scipy.optimize.minimize(f,x0,method='CG',jac=df, options={'maxiter':max_it,"return_all": True})

    imgRicostruita=np.reshape(imgRicostruita_arr.x,(m,n))
    PSNR=metrics.peak_signal_noise_ratio(imgOriginale, imgRicostruita)

    arrPSNR=[]
    arrMSE=[]

    for imgRestoredRaw in imgRicostruita_arr.allvecs:
        imgRestored = np.reshape(imgRestoredRaw, (m, n))
        arrPSNR.append(metrics.peak_signal_noise_ratio(imgOriginale, imgRestored))
        arrMSE.append(metrics.mean_squared_error(imgOriginale, imgRestored))

    return (imgOriginale,imgCorrotta,imgRicostruita,PSNR,MSE,arrPSNR,arrMSE)

def es1():
    (imgOriginale,imgCorrotta,imgRicostruita,PSNR,MSE,arrPSNR,arrMSE)=problemaTest()
    
    plt.figure(figsize=(30,10))

    ax1=plt.subplot(1,2,1)
    ax1.imshow(imgOriginale,cmap="gray")
    plt.title("Immagine originale")

    ax2=plt.subplot(1,2,2)
    ax2.imshow(imgCorrotta,cmap="gray")
    plt.title(f'Immagine corrotta (PSNR: {PSNR: .2f})')

    plt.show()

    plt.figure(figsize=(30,10))

    ax1=plt.subplot(1,2,1)
    ax1.imshow(imgOriginale,cmap="gray")
    plt.title("Immagine originale")

    ax2=plt.subplot(1,2,2)
    ax2.imshow(imgRicostruita,cmap="gray")
    plt.title(f'Immagine ricostruita (PSNR: {PSNR: .2f})')

    plt.show()

    iterazioni=[i for i in range(len(arrPSNR))]

    plt.plot(iterazioni, arrPSNR, label="PSNR")
    plt.plot(iterazioni, arrMSE, label="MSE")
    plt.title("PSNR e MSE al variare delle iterazioni")
    plt.grid()
    plt.legend()

    plt.show()

def es2():
    lambdas = np.linspace(1, 1000, 10)

    allPSNR=[]
    allMSE=[]

    for _ in range(len(lambdas)):
        (imgOriginale,imgCorrotta,imgRicostruita,PSNR,MSE,arrPSNR,arrMSE)=problemaTest()
        allPSNR.append(arrPSNR)
        allMSE.append(arrMSE)

    iterazioni=[i for i in range(len(arrPSNR))]

    fig=plt.figure(figsize=(30,10))

    ax1=plt.subplot(1,2,1)
    for i in range(len(allPSNR)):
        ax1.plot(iterazioni, allPSNR[i], label="PSNR λ="+str(lambdas[i]))
    plt.title("PSNR al variare di lambda")
    plt.grid()
    plt.legend()

    ax2=plt.subplot(1,2,2)
    for i in range(len(allMSE)):
        ax2.plot(iterazioni, allMSE[i], label="MSE λ="+str(lambdas[i]))
    plt.title("MSE al variare di lambda")
    plt.grid()
    plt.legend()

    plt.show()





def main():
    output="Selezionare un numero.\n1: ricostruzione immagine minimi quadrati\n2: ricostruzione immagine Tikhonov(variazioni di λ)\nScelta: "
    scelta=int(input(output))
    
    if(scelta==1): es1()
    elif(scelta==2): es2()
    else: print("Scelta non valida")

main()



# ...

# Di base bisogna:
# - cambiare il problema test
# - si fa il deblur: min{1/2 ||Ax-b||_2^2}
# - grafico di PSNR al variare di lambda
#
# Aggiunta:
# - grafico di PSNR al variare di k
