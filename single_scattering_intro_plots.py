import matplotlib.pyplot as plt
import numpy as np

# lead properties
ALead = 207.2
Na = 6.02E23
X0Lead = 6.37
alpha = 1/137
bohrRadius = 5.29177210903E-11
ZLead = 82


def bremmcross():
    def crossSection(E0, k, Z):
        # define resultant electron beam energy
        E = E0-k
        # define relevant parameters
        b = np.divide(2*E0*E*Z**(1/3), 111*k)
        M0 = (np.divide(k, 2*E0*E)**2+np.divide(Z**(1/3), 111)**2)**-1

        premult = 2*Z**2*bohrRadius**2*alpha*(1/k)
        term1 = (1+np.divide(E, E0)**2-np.divide(2*E, 3*E0)) * \
            (np.log(M0)+1-np.divide(2*np.arctan(b), b))
        term2 = np.divide(E, E0)*(np.divide(2*np.log(1+b**2), b**2) +
                                  np.divide(4*(2-b**2)*np.arctan(b), 3*b**3)-np.divide(8, 3*b**2)+2/9)
        result = premult*(term1+term2)
        return result
    E01 = 50
    E02 = 100
    E03 = 150
    E04 = 200
    E05 = 250

    kArray1 = np.linspace(0, E01, num=100)
    yAxis1 = crossSection(E01, kArray1, ZLead)*kArray1
    kArray2 = np.linspace(0, E02, num=100)
    yAxis2 = crossSection(E02, kArray2, ZLead)*kArray2
    kArray3 = np.linspace(0, E03, num=100)
    yAxis3 = crossSection(E03, kArray3, ZLead)*kArray3
    kArray4 = np.linspace(0, E04, num=100)
    yAxis4 = crossSection(E04, kArray4, ZLead)*kArray4
    kArray5 = np.linspace(0, E05, num=100)
    yAxis5 = crossSection(E05, kArray5, ZLead)*kArray5
    xAxis = kArray1/E01

    fig, ax = plt.subplots(1, 1)
    ax.plot(xAxis, yAxis1, color='b', label='50MeV')
    ax.plot(xAxis, yAxis2, color='orange', label='100MeV')
    ax.plot(xAxis, yAxis3, color='r', label='150MeV')
    ax.plot(xAxis, yAxis4, color='g', label='200MeV')
    ax.plot(xAxis, yAxis5, color='k', label='250MeV')
    ax.grid(True)
    ax.set_xlabel('k/$E_0$')
    ax.set_ylabel('k d$\sigma$/dk')
    plt.legend()


bremmcross()
