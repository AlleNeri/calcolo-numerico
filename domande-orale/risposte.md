# Condizionamento
Il condizionamento di una matrice ci dice quanto una perturbazione sui dati può influenzare l'errore sul risultato.

# Errore
L'errore è calcolato in norma 2 come: $$\frac{ \| x^*-x \| }{ \| x \| }$$

# Metodi iterativi
I metodi iterativi stazionari, quali Jacobi e Gauss-Seidel, si basano sulla formula: $$x_k=Tx_{k-1}+c$$
Per ottenere la matrice $T$, detta matrice di iterazione, e $c$ si scompone $A$ in: $$A=M-N$$
I metodi iterativi stazionari si differenziano per come ottengono le matrici $M$ e $N$. 
Si parte dalla scomposizione della matrice data $A$ in: $$A=D-E-F$$
Da questa scomposizione derivano le matrici di iterazione:
- per Jacobi la matrice di iterazione è: $J=D^{-1}(E+F)=I-D^{-1}A$
- per Gauss-Seidel la matrice di iterazione è: $L_1=(D-E)^{-1}F$

# Definizione dei metodi iterativi
Vedi [spiegazione scomposizione dei metodi iterativi](#metodi-iterativi).
Il metodo di Jacobi è definito se la matrice $D$ è non singolare, ovvero invertibile.
Il metodo di Gauss-Seidel è definito se le matrice $D-E$ è non singolare, ovvero invertibile.
Queste due condizioni sono entrambe soddisfatte se la diagonale principale della matrice $A$ non contiene $0$.

# Convergenza dei metodi iterativi
Vedi [spiegazione scomposizione dei metodi iterativi](#metodi-iterativi).
I metodi iterativi convergono se e solo se la loro matrice di iterazione ha un raggio spettrale inferiore a 1.

# Scomposizione SVD
Risolvere un sistema $Ax=b$, con $A\in\mathbb{R}^{m,n}$, $m>n$, sovradimensionato è un problema che non sempre presenta risultato esatto.
È possibile avere un risultato solo se la matrice $A$ ha rango massimo.
In questo caso la soluzione del sistema si ottiene dall'equazione notrmale: $$A^TAx=A^Tb$$
Altrimenti si può stabilire come soluzione del sistema il valore di $x$ che rende il vettore residuo minimo.
Il valore residuo è la quantità: $$r=Ax-b$$
Si ha quindi che la soluzione sarà:
$$min_{x\in\mathbb{R}^n}\|r\|_ 2^2=min_{x\in\mathbb{R}^n}\|Ax-b\|_2^2$$

Per ottenere la soluzione di norma minima si effettua la scomposizione della matrice $A$: $$A=U \Sigma V^T$$
- $\Sigma\in\mathbb{R}^{m,n}$ è la matrice che ha nelle prime $n$ righe una matrice diagonale i cui coefficenti sono $\sigma_1\geq\dots\geq\sigma_n\geq0$ i valori singolari di $A$ e nelle restanti righe vettori nulli.
- $U\in\mathbb{R}^{m,m}$ è la matrice che ha per colonne i vettori singolari sinistri di $A$.
- $V\in\mathbb{R}^{n,n}$ è la matrice che ha per colonne i vettori singolari destri di $A$.

Da questa scomposizione si può ottenere la pseudoinversa della matrice $A$:

$$A^+=U \Sigma^+ V^T \text{ con } \Sigma^+_{i,j} = \begin{cases} \frac{1}{\sigma_i} & \text{ se } i=j \text{ e } i\leq k \\
0 & \text{altrimenti} \end{cases}$$

Con essa si può risolvere il sistema come:
$$x^*=A^+b$$

Per poter iterare su questa formula si porta nella forma:
$$x^*=\Sigma_{i=1}^{rank(A)}\frac{u_i^Tb}{\sigma_i}v_i$$

# Forma diadica
Vedi [spiegazione scomposizione SVD](#scomposizione-svd).
Scomponendo una matrice $A$ col metodo SVD la matrice poi può essere scritta come somma di diadi(matrici di rango 1).
La forma diadica della matrice $A$ è: $$A=\Sigma_{i=1}^{rank(A)}\sigma_i u_i v_i$$

# Zeri di funzione
Si tratta di trovare, all'interno di un intervallo $[a,b]$ finito, l'intersezione di una funzione $f(x)$ con l'asse x.
Se la funzione assume segni diversi nei punti $a$ e $b$, quindi $f(a)f(b)<0$, ed è continua nell'intervallo $[a,b]$ si ha che $f$ interseca sicuramente l'asse x nell'intervallo $[a,b]$.

# Metodo della Bisezione
È un metodo per individuare lo zero di una funzione.
Vedi [spiegazione degli zeri di funzione](#zeri-di-funzione).
Per questo metodo si è in grado di individuare un numero di iterazioni minimo, un lower bound.
Il numero di iterazioni minimo data la tolleranza $\delta$ è calcolato come: $$k \geq \log_2\frac{b-a}{\delta}$$
Si procede iterativamente a dividere l'intervallo, che all'iterazione $k$ sarà, $[a_k,b_k]$ a metà finchè non si individua il valore di $x$ che annulla la funzione $f(x)$.
Si inizia calcolando il punto medio $c_k$ tra $a_k$ e $b_k$: $$c=\frac{a_k+b_k}2$$.
Se la distanza fra $f(c_k)$ e $0$ è mionre della tolleranza il risultato ottenuto è accettabile.
Altrimenti si sono ottenuti 2 segmenti: $[a_k,c_k]$ e $[c_k,b_k]$.
Solo uno dei 2 segmenti può avere la funzione di segno opposto agli estremi, esso sarà selezionato e prenderà il posto dell'intervallo $[a_{k+1},b_{k+1}]$ all'inizio dell'iterazione successiva, $k+1$.
Per individuare quale segmento della bisezione è da selezionare per l'iterazione $k+1$ si valutano i segni della funzione:
- se $f(a_k)$ e $f(c_k)$ hanno segno opposto il $b_{k+1}=c_k\ \ \Rightarrow\ \ [a_{k+1},b_{k+1}]=[a_k,c_k]$.
- altrimenti $f(b_k)$ e $f(c_k)$ hanno segno opposto quindi $a_{k+1}=c_k\ \ \Rightarrow\ \ [a_{k+1},b_{k+1}]=[c_k,b_k]$.

# Approssimazioni successive o di punto fisso
È un metodo per individuare lo zero di una funzione.
Vedi [spiegazione degli zeri di funzione](#zeri-di-funzione).
Data la funzione $f(x)$ si risolve il problema $f(x)=0$ indirettamente risolvendo il problema di punto fisso, $g(x)=x$, di un'altra funzione: $g(x)$.
La funzione $g(x)$ si ottiene da quella data come: $$g(x)=x-f(x)\phi(x)$$
Si ha sulla funzione $\phi(x)$ il vincolo: $$0< |\phi(x)| <\infty,\ x\in[a,b]$$
Si ottiene quindi un metodo iterativo riassunto nalla formula: $$x_{k+1}=g(x_k)$$

# Metodo di Newton
Il metodo di Newton è un'istanza del metodo delle approssimazioni successive.
Vedi [spiegazione del metodo delle approssimazioni successive](#approssimi-successive-o-di-punto-fisso).
La particolarità del metodo di Newton per il metodo delle approssimazioni successive è quella di individuare come funzione $\phi(x)$ l'inversa della derivata prima della funzione $f(x)$.
Si ha come risultato che $g(x)$ vale: $$g(x)=x-\frac{f(x)}{f'(x)}$$
La formula di questo metodo iterativo diventa: $$x_{k+1}=x_k-\frac{f(x_k)}{f'(x_k)}$$
Si nota che la derivata di $f(x)$ non può essere nulla per $x_k$: $$f(x_k)\neq 0$$

# Minimi di funzione
Il probelma di minimizzazione non vincolata è quello di individuare il punto minimo di una funzione. In particolar modo abbiamo applicato questo problema alle funzioni, anche detti campi scalari, del tipo $f:\mathbb{R}^2\to\mathbb{R}$; la cui rappresentazione grafica è una superfice.
Si distinguono diversi punti di minimo:
- punti di minimo locale di $f$: $x^* $ tale che esista un $\varepsilon>0$ per cui $f(x^* )\leq f(x) \ \forall\  x$ tale che $\|x-x^* \|<\varepsilon$.
- punti di minimo locale in senso stretto di $f$: $x^* $ tale che esista un $\varepsilon>0$ per cui $f(x^* )\ <\ f(x) \ \forall\  x$ tale che $\|x-x^* \|<\varepsilon$, $x\neq x^* $.
- punti di minimo globale di $f$: $x^* $ tale per cui $f(x^* )\leq f(x) \ \forall\  x\in\mathbb{R}^n$.
- punti di minimo globale in senso stretto di $f$: $x^* $ tale per cui $f(x^* )\ <\ f(x) \ \forall\  x\in\mathbb{R}^n$, $x\neq x^* $.

Per risolvere questo tipo di problema esitono i metodi di discesa.
I criteri di arresto di questi mietodi possono essere di diversa natura: raggiungimento della soluzione, con la dovuta tolleranza, o fallimento del metodo.
I criteri di arresto sono quindi:
- $\|\nabla f(x_k)\|\leq \varepsilon_1$ dove $\varepsilon_1$ è la tolleranza stabilita.
- $\|x_k-x_{k-1}\|<\varepsilon_2$ dove $\varepsilon_2$ è la tolleranza stabilita.

Questi metodi si dividono in 2 categorie: line search e trust region.
La categoria line serach prevede la generazione di una successione del tipo: $$x_{k+1}=x_k+\alpha_k p_k\ \ \ \text{ dove }\ \ \ p_k\in\mathbb{R}^n,\alpha_k\in\mathbb{R}^+$$
I parametri $p_k$ e $\alpha_k$ sono scelti per garantire il decremento della funzione obiettivo $f$:
$$f(x_{k+1}) < f(x_k)\ \forall\ k$$
Stabilire questi parametri è il modo in	cui di definisce la convergenza del metodo e la sua rapidità.
In generale la scelta della direzione, quindi di $p_k$, determina la rapidità di convergenza; mentre la scelta del passo, quindi di $\alpha_k$, garantisce la convergenza.
Anche se valori poco appropriati di $\alpha_k$ potrebbero determinare un rallentamento della convergenza del metodo.
Il sottoproblema di scegliere il passo è detto ricerca in linea(line search) o ricerca unidirezionale in quanto si ha una direzione prestabilita, ovvero quella di $p_k$.
La ricerca lineare esatta non può essere compiuta in maniera esatta in quanto è computazionalmente costosa.
In sua vece ne viene compiuta una inesatta che necessita però di alcuni accorgimenti.
Si ha infatti che la scelta di $\alpha_k$ deve far in modo che le condizioni di arresto non siano verificate.
Per questo si hanno le condizioni di Wolfe:
- condizione di Armijo: assicura una decrescita sufficiente. $f(x_k+\alpha p_k)\leq f(x)+c_1\alpha\nabla f(x_k)^T p_k$
	- essa non è sufficiente a garantire la decrescita, ma se combinara al backtraking lo diventa.
- condizione di curvatura: assicura che il passo non sia troppo piccolo. $\nabla f(x_k +\alpha_k p_k)^T\geq c_2\nabla f(x_k)^T p_k$

L'algoritmo di backtraking pone un valore $\alpha=\bar{\alpha}$, solitamente 1, che diminusice fino a soddisfatte la condizione di Armijo.
Il fattore di decremento di questo metodo è $\rho$, si ha quindi che: $\alpha=\rho\alpha$.
Nella pratica si sceglie un valore $\rho=\cfrac{1}{2}$.

# Metodo di discesa rapida
È uno dei metodi più semplice per il problema del minimo di un campo scalare.
Fa parte dei metodi di discesa e della categoria line search.
Ha un basso costo computazionale per iterazione, ma converge molto lentamente quindi ha un costo complessivo alto.
In breve esso sceglie:
- $p_k=-\nabla f(x_k)\ \forall\ k$.
- $\alpha$ con il metodo cella ricerca lineare esatta.

# Metodo del gradiente
Il metodo del gradiente è un metodo per ottenere il minimo di un campo scalare utilizzando appunto il gradiente o operatore lambda delle derivate parziali.
Fa parte dei metodi di discesa e della categoria line search.
In breve esso sceglie:
- $p_k=-\nabla f(x_k)\ \forall\ k$.
- $\alpha$ con il metodo cella ricerca lineare inesatta, con l'algoritmo del backtraking.

Si può dimostrare che la convergenza del metodo del gradiente è lineare.

# Metodo di Newton puro
Il metodo di Newton puro è un metodo per ottenere il minimo di un campo scalare.
Esso sfrutta come direzione di ricerca la direzione di Newton: $p_k=-H_f(x_k)^{-1}\nabla f(x_k)$ con $H_f(x_k)$ matrice Hessiana, ovvero $\nabla^2 f(x_k)$.
Mentre il passo vale: $\alpha_k=1\ \forall\ k$.
La convergenza del metodo è fortemente legata alla proprietà della matrice Hessiana di essere definita positiva; quindi non è possibile dare una considerazione generale.

# Deblur di immagini
Un'immagine ha 2 sorgenti di degradazione:
- il processo di formazione dell'immagine(blurring).
- il processo di misurazione dell'immagine(noise).
Il probelma è quello di ricostruire una buona approssimazione dell'oggetto reale partendo dall'immagine acquisita, degradata da blur e rumore, e da qualche informazione sulle degradazioni.

Per quanto riguarda le informazioni sulle degradazioni se ne ha una importante dalla rilevazione di un'immagine con un solo pixel, essa è detta Single Pixel Image(SPI), dalla quale si ottiene la Point Spread Function.

---
# Fattorizzazione LR
## Numero di condizione 1
Vedi [spiegazione condizionamento](#condizionamento).
Il condizionamento sulla matrice A non è legato alla sua dimensione. Si nota infatti sul grafico come la matrice casuale A abbia picchi di condizionamento a valori casuali e non prevedibili(eseguendo una seconda volta il codice si noterà come i picchi muteranno).
## Soluzione del sistema con fattorizzazione LR con pivoting
La validità della soluzione è rilevabile dal suo errore.
Vedi [spiegazione errore](#errore).
Si nota dal grafico che l'errore è slegato dalla dimensione della matrice A.
##Condizionamento e soluzione del sistema
Confrontando i due grafici si nota che i picchi di uno corrisponodono a quelli dell'altro.
È possibile riscontrare ciò inserendo nello stesso grafico le due curve, ma per accorgersi della corrispondenza dei picchi di una con quelli dell'altra si necessita di portare le due misure a un ordine di grandezza confrontabile: essi distano di un ordine di grandezza di circa $10^{16}$.
È possibile trovare questa corrispondenza in quanto le due quantità sono legate dalla definizione di condizionamento.
Vedi [spiegazione condizionamento](#condizionamento).

# Fattorizzazione di Cholesky con matrice di Hilbert
## Numero di condizione
Vedi [spiegazione condizionamento](#condizionamento).
Il condizionamento sulla matrice di Hilbert è alto quindi si nota che esso "esplode" per una dimensione della matrice relativamente bassa come 15.
## Soluzione del sistema con fattorizzazione di Cholesky
La soluzione è valutata tramite l'errore.
Vedi [spiegazione errore](#errore).
Può sembrare che l'errore creasca con la crescita della dimensione della matrice, ma non è così in quanto la sua crescita risulta legata al condizionamento della matrice.
## Condizionamento e soluzione del sistema
Confrontando i due grafici si nota che il picco di condizionamento nel primo grafico corrisponde a quello dell'errore nel secondo.
È possibile trovare questa corrispondenza in quanto le due quantità sono legate dalla definizione di condizionamento.
Vedi [spiegazione condizionamento](#condizionamento).

# Fattorizzazione di Cholesky con matrice tridiagonale
## Numero di condizione
Vedi [spiegazione condizionamento](#condizionamento).
Si nota che il condizionamento sulla matrice tridiagonale si stabilizza a un valore relativamente basso(quello della matrice di Hilbert ha ordine di grandezza di $10^{18}$) col crescere della dimensione della matrice.
## Soluzione del sistema con fattorizzazione di Cholesky
La soluzione è valutata tramite l'errore.
Vedi [spiegazione errore](#errore).
L'errore della soluzione, nonostante alcune oscillazioni, risulta decrescere in quanto la matrice tridiagonale è ben condizionata.
## Condizionamento e soluzione del sistema
Confrontando i due grafici si nota che la crescita del condizionamento nel primo grafico corrisponde a un calo dell'errore nel secondo.
Questo è dato dal fatto che, nonostante ci sia una cresita di condizionamento, quat'ultimo si stabilizza è per ciò che anche l'errore si va a stabilizzare in difetto.
È possibile trovare questa corrispondenza in quanto le due quantità sono legate dalla definizione di condizionamento.
Vedi [spiegazione condizionamento](#condizionamento).

# Metodi iterativi con dimensione fissa
## Controlli
Vedi [spiegazione definizione dei metodi iterativi](#definizione-dei-metodi-iterativi).
Vedi [spiegazione Convergenza dei metodi iterativi](#convergenza-dei-metodi-iterativi).
## Soluzione del sistema con metodi iterativi
La soluzione è valutata tramite l'errore.
Vedi [spiegazione errore](#errore).
### Dinensione fissa della matrice
Si nota come entrambe i metodo convergono per la matrice tridiagonale e che il numero di iterazioni non raggiunge il massimo stabilito.
Ciò è segno che i metodi si arrestano per aver raggiunto la tolleranza stabilita.
Si evidenzia che il metodo di Jacobi, pur partendo con un errore nelle prime iterazioni minore a quello di Gauss-Seidel, si arresta parecchio dopo a Gauss-Seidel.
### Dimensione variabile della matrice
L'errore del metodo di Jacobi, rispetto a quello del metodo di Gauss-Seidel, pur partendo con valori non distanti, si impenna immediatamente.
Si ha inoltre che, tolte, se pur sigificative le oscillazioni; si può notare una stabilizzazione dell'errore.
L'errore inoltre rientra nell'orfine di grandezza definito.
## Numero di iterazioni
Si premette che il numero di iterazioni ha un limite superiore definito dal massimo numero di iterazioni permesse dai metodi.
Si nota facilmente che il numero di iterazioni del metodo di Jacobi sono superiori a quelle del metodo di Gauss-Seidel.
Nonostante ciò i due metodi sembrano seguire uno stesso andamento logaritmico, che quindi si stabilizza.

# Confronto dei metodi
Si fa riferimento alla matrice tridiagonale.
## Errore dei metodi
Vedi [spiegazione errore](#errore).
L'errore dei metodi iterativi risulta più alto di quello dei metodi di fattorizzazione LU e di Cholesky.
Questo perchè i primi sono un troncamento di un operazione infinita.
I metodi di fattorizzazione producono invece un risultato esatto.
## Tempo di esecuzione dei metodi
Premessa: la rilevazione sui tempi potrebbe essere falsata da alcune dinamiche di scheduling non prevedibili.
Si notano alcune oscillazioni nei tempi di esecuzione dei metodi di fattorizzazione; quelle maggiori sono nella farrorizzazione LU.
Il grafico sottolinea anche la bassa crescita del tempo delle fattorizzazioni LU e di Cholesky rispetto a quelle dei metodi iterativi.
Ciò porta a dire che, nonostante sembrino inizialmente confrontabili i tempi dei metodi iterativi rispetto a quelli di fattorizzazione, col crescere della dimensione della matrice sono più convenienti i tempi dei metodi di fattorizzazione in generale e di Cholesky in particolare.
## Errore e tempi dei metodi a confronto
Sia le valutazioni degli errori che quelle dei tempi danno come conveniente i metodi stazionari.

# Regressione polinomiale ai minimi quadrati
## Metodo di scomposizione SVD
Vedi [spiegazione scomposizione SVD](#scomposizione-svd).
Il metodo di scomposizione SVD può essere utilizzato per calcolare una regressione polinomiale di un dataset: ottenere da un insieme di coordinate un polinomio che ne approssimi l'andamento.
Per fare ciò, considerato il dataset $(x_i,y_i)\ \forall i\in\{1,\dots,m\}$ e deciso il grado del polinomio come $n$, si necessita di creare la matrice $A$.
Essa avrà come colonne il vettore $x$ i cui elementi vengono elevati a uno in meno del numero di colonna per un totale di $n+1$ colonne.
In definitiva si otterrà: $A\in\mathbb{R}^{m,n+1}$
Da questa matrice $A$ si può applicare il metodo di risoluzine per il sistema sovradimensionato $A\alpha=y$ Risolvibile sia con il metodo di scomposizione SVD sia con le equazioni normali.
## Soluzione con le equazioni normali
Vedi [spiegazione scomposizione SVD](#scomposizione-svd).
Questo metodo è percorribile solo quando la matrice $A$ ha rango massimo.
## Soluzione con la scomposizione SVD
Vedi [spiegazione scomposizione SVD](#scomposizione-svd).
Il metodo è sempre percorribile.
## Confronto dei polinomi
Si nota che i polinomi al crescere del grado approssimano sempre meglio i punti del dataset.
Inoltre i due metodi danno polinomi molto simili, quasi non distinguibili.

# Compressione di immagini
È possibile comprimere le immagini utilizzando la scomposizione SVD.
Ogni canale di colore di un'immagine è come una matrice; considerando essa la matrice $A$ della scomposizione in diadi si ottiene una compressione dell'immagine.
Vedi [spiegazione forma diadica](#forma-diadica).
## Errore relativo
L'errore relativo dell'immagine si calcola tramite la matrice $A$, che rappresenta l'immagine stessa, e la matrice $A_p$, che rappresenta l'immagine ricostruita, tramite il calcolo dell'errore per vettori.
Vedi [spiegazione errore](#errore).
L'errore relativo dell'immagine risulta discendere col crescere del numero di diadi che si vanno ad aggiungere al ricomposizione dell'immagine.
Si ottiene però che l'errore relativo placa gradualmente la sua discesa; questo è dato dal fatto che sono le primissime diadi a dare, anche visivamente il maggiore apporto per il riconoscimento dell'immagine stessa.
## Fattore di compressione
Il fattore di compressione è calcolato come: $$c_p=\frac{min(m,n)}p -1$$
Questo parametro da l'idea allo stesso tempo di quanta informazione viene persa e di quanto viene alleggerita l'immagine.
Anche il fattore di compressione cala al crescere delle diadi aggiunte, che apportano più informazioni, ma anche pesantezza all'immagine.
Esso risente molto di più rispetto all'errore dell'aggiunta di diadi, infatti la sua discesa è molto più brusca.
Mantiene però come l'errore una certa stabilità dopo un certo numero di diadi perchè come detto sono le pime ad apportare maggiori informazioni all'immagine.

# Calcolo degli zeri di funzione
Per il problema del calcolo degli zeri di funzione si utilizzano i metodi di bisezione, di Newton e delle approssimazioni successive o di punto fisso.
Vedi spiegazione del [problema di calcolare gli zeri di una funzione](#zeri-di-funzione), del [metodo di bisezione](#metodo-della-bisezione), del [metodo di Newton](#metodo-di-newton) e del [metodo delle approssimazioni successive o di punto fisso](#approssimazioni-successive-o-di-punto-fisso).
## Numero di iterazioni
Il numero di iterazioni nel metodo di bisezione vede un lower bound calcolabile prima dell'inizio delle iterazioni.
Quel valore è il numero minimo di iterazioni per raggiungere la tolleranza proposta.
Questo numero si aggira attorno alla trentina per le funzioni date e l'errore stabilito.
Nel metodo di punto fisso, e quindi anche nel metodo di Newton, si stabilisce un numero massimo di iterazioni, il cui raggiungimento è da aggiungere alle condizioni di arresto della funzione.
Queste ultime permettono di terminare le iterazioni se si raggiunge un ordine di grandezza dell'errore accettabile.
Si raggiunge lerrore accettabile solo nella terza funzione.
## Errore
L'errore che il metodo di bisezione commette è oscillante perchè, considerando i punti medi, non si cura se l'errore aumenta o diminusice nel corso delle iterazioni.
Il metodo di punto fisso, e quindi anche nel metodo di Newton, a ogni iterazione migliorano l'approssimazione, non a caso si dice anche metodo delle approssimazioni successive.
In questo moetodo l'errore viene coinvolto anche nei criteri d'arresto; in particolar modo sia l'errore relativo che quello assoluto.
Il primo per per assicurarsio che si stia effettivamente miglirando il risultato a ogni iterazione; il secondo per non superare la tolleranza stabilita.

# Calcolo del minimo di una funzione
Per risolvere il problema del calcolo dei minimi di una funzione si utilizza il metodo del gradiente.
Vedi [spiegazione dei minimi di funzioni](#minimi-di-funzioni) e [spiegazione del metodo del gradiente](#metodo-del-gradiente).
Si risolve il problema sia con il meodo a passo fisso che con quello a passo variabile.
In generale quello a passo fisso è meno costoso a livello computazionale di quello a passo variabile che deve applicare una tecnica di backtraking.
Ma si nota come il numero di iterazioni sia più alto a fronte di risultati generalmente peggiori nel metodo con passo fisso.
Ciò si nota in particolar modo nella prima funzione.
## Errore
Si nota che l'errore della prima funzione è più elevato quando si utilizza il metodo con step fisso.
In generale si ha che l'errore relativo si riduce logaritmicamente al variare delle iterazioni.
Si ha questo in quanto ogni iterazione approssima sempre maggiormente il valore del minimo del campo scalare.
## Valore della funzione
Si nota, nella prima funzione in particolar modo, che il valore del campo scalare è afflitto da un errore maggiore quando si utilizza il metodo a passo fisso.
In generale si ha comunque un decremento della funzione che si attesta, con lievi cambiamenti, a un valore preciso, quello di minimo.
## Norma del gradiente
La norma del gradiente è il criterio di terminazione che viene raggiunto più di frequente, in particolar modo quando si impiega il metodo a passo variabile.
Il suo valore è infatti nell'ordine di $10^-5$ nei grafici del metodo con passo fisso e non riesce a raggiungere quell'ordine di grandeza nell'alto metodo, in particolar modo nella prima funzione.
