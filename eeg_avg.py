import numpy as np
import matplotlib.pyplot as plt

def simulate_ERP(fs: int=250, 
                 latency: int=100,
                 N_exp: int=5,
                 plotting: bool=True,
                 vary: str='homogenous') -> np.ndarray:
    """
    Simulates a series of ERP EEG readings, with varying amplitude and/or variance

    Args:
        fs (int, optional): sampling frecuency. Defaults to 250
        latency (int, optional): latency of ERP. Defaults to 100.
        N_exp (int, optional): Number of independent experiments. Defaults to 5.
        plotting (bool, optional): Indicates wheteher or not to plot the EEGs. Defaults to True.
        mode (str, optional): Indicates how to vary either amplitude of noise variance:
            - 'homogenous': same amplitude and variance for all experiments
            - 'amp': varies amplitude with every experiment
            - 'var': varies noise variance for every experiment
            - 'both': vary both amplitude and noise variance
        Defaults to 'homogenous'

        Returns: X, a matrix where every row is a new experiment and every column is a new sample

    """
    lat = latency*(10**-3)*fs
    t_ERP = np.linspace(0, 0.2, int(fs*0.2))

    VALID_VARY = {'homogenous', 'amp', 'var', 'both'}
    if vary not in {'homogenous', 'amp', 'var', 'both'}:
        raise ValueError(F"{vary} is not a valid mode. Should be: {''.join(VALID_VARY)}")

    elif vary == 'homogenous' or vary == 'var':
        erp = 20*np.sin(100*t_ERP)*np.exp(-30*t_ERP)
        erp = erp/np.max(erp)
        l = np.zeros([1, int(lat)])
        s = np.append(l, erp)
        end = np.zeros([1, fs-s.shape[0]])
        sig = np.append(s, end)
        erp_signal = sig.copy()
        eeg = []
        for i in range(N_exp):
            eeg.append(erp_signal)
        eeg = np.array(eeg)

    elif vary == 'amp' or vary == 'both':
        eeg = []
        for i in range(N_exp):
            erp = (5+5*abs(np.random.randn(1)[0])) * np.sin(100*t_ERP)*np.exp(-30*t_ERP)
            l = np.zeros([1, int(lat)])
            s = np.append(l, erp)
            end = np.zeros([1, fs-s.shape[0]])
            sig = np.append(s, end)
            eeg.append(sig)
        eeg = np.array(eeg)
        eeg = eeg/(0.1*np.max(eeg))
        erp_signal = np.mean(eeg, axis=0)

    if vary == 'homogenous' or vary == 'amp':
        for i in range(N_exp):
            noise = 10*np.random.randn(sig.shape[0])
            eeg[i,:] = eeg[i,:] + noise
    elif vary == 'var' or vary == 'both':
        for i in range(N_exp):
            noise = (5+3*abs(np.random.randn(1)[0])) * np.random.randn(sig.shape[0])
            eeg[i,:] = eeg[i,:] + noise

    t_eeg = np.linspace(0, eeg.shape[1]/fs, eeg.shape[1])

    if plotting:
        plt.figure(figsize=(10,5))
        for i in range(N_exp):
            plt.plot(t_eeg, eeg[i]-i*50)
        plt.title('Realizaciones')
        plt.ylabel('Amplitud [uV]')
        plt.xlabel('Tiempo [s]')
        plt.show()

    return t_eeg, eeg, erp_signal

def average_EEG(X: np.ndarray, mode: str='homgenous') -> np.ndarray:
    """
    Performs a weighted or unweighted average of series of ERP EEG signals

    Args:
        X (np.ndarray): NxM matrix where every row is a new experiment and every column is a new sample
        mode (str, optional): Indicates how to perform the average. Could be:
            - homogenous: simple, unweighted average
            - amp: weight by amplitude
            - var: weight by variance
            - both: weight by both amplitude and variance
        Defaults to 'homogenous'.

    Returns:
        np.ndarray: an Mx1 array with the averaged signals
    """
    VALID_MODE = {'homogenous', 'amp', 'var', 'both'}
    if mode not in {'homogenous', 'amp', 'var', 'both'}:
        raise ValueError(F"{mode} is not a valid mode. Should be: {''.join(VALID_MODE)}")

    elif mode == 'homogenous':
        return np.mean(X, axis=0)
    
    # Find amplitudes
    s = np.mean(X, axis = 0)
    a = X.dot(s.T)

    # Find variances
    M = X.shape[1]
    V = np.var(X[:, -int(0.4*M):], axis=1)

    # Get weights and average
    if mode == 'amp':
        w = a / np.sum(a**2)
    elif mode == 'var':
        w = (1/V) / (np.sum(1/V))
    elif mode == 'both':
        w = (a/V) / (np.sum(a**2/V))
    
    return w.T.dot(X/np.sum(w))


def test():
    t, X, pe = simulate_ERP(N_exp = 200, vary='both' , plotting=False)

    X_mean = average_EEG(X, mode='homogenous') 
    X_amp = average_EEG(X, mode='amp')
    X_var = average_EEG(X, mode='var')
    X_both = average_EEG(X, mode='both')

    plt.figure(figsize = (20,10))
    plt.plot(t, X_mean, label='Promediado homogeneo')
    plt.plot(t, X_amp, label='Promedio inhomogéneo amplitud variable')
    plt.plot(t, X_var, label='Promedio homogéneo varianza variable')
    plt.plot(t, X_both, label='Promedio inhomogéneo todo variable')
    plt.plot(t, pe, color='black', label='Potencial Evocado teorico')
    plt.title('Métodos de promediado')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test()