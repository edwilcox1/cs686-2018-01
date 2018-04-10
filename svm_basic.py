
from classifier import classifier

'''#######********************************
Non-Kernel VErsions below
#######********************************
'''
import numpy as np

from numpy import mat, zeros, multiply, shape, nonzero, random


class _SVMStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):  # Initialize the structure with the parameters
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2)))  # first column is valid flag



class svm_basic(classifier):

    def __init__(self, C=1.0, toler=0.001, maxIter=50):
        self.C = C
        self.toler = toler
        self.maxIter = maxIter
        self.weights = None
        self.alphas = None
        self.b = None

    def fit(self, X, Y):
        x = mat(X)
        y = mat(Y)
        oS = _SVMStruct(x, y.transpose(), self.C, self.toler)
        self.b, alphas = self.__smoPK(oS, self.maxIter)
        self.__calc_weights(x, y, self.b, alphas)
        return self.weights, self.b

    def predict(self, X):
        return X * self.weights + self.b

    def __calc_weights(self, x, y, b, alphas):
            y = y.T
            m, n = shape(x)
            self.weights = zeros((n, 1))
            for i in range(m):
                self.weights += multiply(alphas[i] * y[i], x[i,:].T)


    def __updateEkK(self, oS, k):  # after any alpha has changed update the new value in the cache
        Ek = self.__calcEkK(oS, k)
        oS.eCache[k] = [1, Ek]

    def __selectJK(self, i, oS, Ei):  # this is the second choice -heurstic, and calcs Ej
        maxK = -1
        maxDeltaE = 0
        Ej = 0
        oS.eCache[i] = [1, Ei]  # set valid #choose the alpha that gives the maximum delta E
        validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
        if (len(validEcacheList)) > 1:
            for k in validEcacheList:  # loop through valid Ecache values and find the one that maximizes delta E
                if k == i: continue  # don't calc for i, waste of time
                Ek = self.__calcEkK(oS, k)
                deltaE = abs(Ei - Ek)
                if (deltaE > maxDeltaE):
                    maxK = k;
                    maxDeltaE = deltaE;
                    Ej = Ek
            return maxK, Ej
        else:  # in this case (first time around) we don't have any valid eCache values
            j = self.__selectJrand(i, oS.m)
            Ej = self.__calcEkK(oS, j)
        return j, Ej

    def __innerLK(self, i, oS):
        Ei = self.__calcEkK(oS, i)
        cond1 = oS.labelMat[i] * Ei < -oS.tol
        cond2 = oS.alphas[i] < oS.C
        cond3 = oS.labelMat[i] * Ei > oS.tol
        cond4 = oS.alphas[i] > 0
        # if ((oS.labelMat[i] * Ei) < -oS.tol) and (oS.alphas[i] < oS.C) or \
        #         ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        if cond1 and cond2 or (cond3 and cond4):
            j, Ej = self.__selectJK(i, oS, Ei)  # this has been changed from selectJrand
            alphaIold = oS.alphas[i].copy()
            alphaJold = oS.alphas[j].copy()
            if (oS.labelMat[i] != oS.labelMat[j]):
                L = max(0, oS.alphas[j] - oS.alphas[i])
                H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
            else:
                L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
                H = min(oS.C, oS.alphas[j] + oS.alphas[i])
            if L == H: print("L==H"); return 0
            eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
            if eta >= 0: print("eta>=0"); return 0
            oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
            oS.alphas[j] = self.__clipAlpha(oS.alphas[j], H, L)
            self.__updateEkK(oS, j)  # added this for the Ecache
            if abs(oS.alphas[j] - alphaJold) < 0.00001: print("j not moving enough"); return 0
            oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (
            alphaJold - oS.alphas[j])  # update i by the same amount as j
            self.__updateEkK(oS, i)  # added this for the Ecache                    #the update is in the oppostie direction
            b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T \
                      - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
            b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T \
                 - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
            if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
                oS.b = b1
            elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
                oS.b = b2
            else:
                oS.b = (b1 + b2) / 2.0
            return 1
        else:
            return 0

    def __selectJrand(self, i, m):
        j = i  # we want to select any J not equal to i
        while j == i:
            j = int(random.uniform(0, m))
        return j

    def __clipAlpha(self, aj, H, L):
        if aj > H:
            aj = H
        if L > aj:
            aj = L
        return aj

    def __calcEkK(self, oS, k):
        # intermediate = multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)
        # fXk = np.float64(intermediate) + oS.b
        fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
        Ek = fXk - float(oS.labelMat[k])
        return Ek

    def __smoPK(self, oS, maxIter):  # full Platt SMO
        iter = 0
        entireSet = True
        alphaPairsChanged = 0
        while iter < maxIter and (alphaPairsChanged > 0 or entireSet):
            alphaPairsChanged = 0
            if entireSet:  # go over all
                for i in range(oS.m):
                    alphaPairsChanged += self.__innerLK(i, oS)
                    print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
                iter += 1
            else:  # go over non-bound (railed) alphas
                nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < self.C))[0]
                for i in nonBoundIs:
                    alphaPairsChanged += self.__innerLK(i, oS)
                    print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
                iter += 1
            if entireSet:
                entireSet = False  # toggle entire set loop
            elif alphaPairsChanged == 0:
                entireSet = True
            print("iteration number: %d" % iter)
        return oS.b, oS.alphas

