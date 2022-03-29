from math import hypot, ceil
from PySide2 import QtGui
import numpy as np
from math import pi, cos, tan
from scipy.special import jv  # Bessel function
from itertools import product
import matplotlib.pyplot as plt

#Qt imports
from PySide2.QtCore import QDateTime
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
# from PyQt5.uic import loadUi

import sys

try:
    from KernelGeneratorGUI import *
except:
    print("Can't import KernelGeneratorGUI file")

class memoize:

    """ Memoization decorator to cache repeatedly-used function calls """

    # stock code from http://avinashv.net

    def __init__(self, function):
        self.function = function
        self.memoized = {}

    def __call__(self, *args):
        try:
            return self.memoized[args]
        except KeyError:
            self.memoized[args] = self.function(*args)
            return self.memoized[args]


@memoize
def lommel(n, a, b):

    """ Calculates the nth lommel function """

    U = 0
    for k in range(0, 100000):
        sum = ((-1)**k * (a/b)**(n+2*k) * jv(n+2*k, pi*a*b))
        U += sum
        if abs(sum) < 0.00001:
            return U
    raise ValueError("Failure to converge")


@memoize
def generatePoints(starR):

    """ Models star as an array of uniformly distributed point sources """

    if starR == 0:  # model as point source
        return np.array([(0,0)])
    n = 5  # number of points to model 1D radius of star
    pairs = np.array([item for item in product(np.linspace(-starR, starR, 2*n-1), repeat=2) if hypot(item[0], item[1]) <= starR])
    return pairs


def diffractionCalc(r, p, starR, lam, D, b):

    """ Analytically calculates intensity at a given distance from star centre """

    # r is distance between line of sight and centre of the disk, in fresnel scale units
    # p is radius of KBO, in fresnel scale units
    pts = generatePoints(starR)
    r = fresnel(r, lam, D)
    res = 0
    effR = np.round(np.hypot((r - pts[:, 0]), (pts[:, 1]-b)), 2)
    coslist = np.cos(0.5*pi*(effR**2 + p**2))
    sinlist = np.sin(0.5*pi*(effR**2 + p**2))
    l = len(pts)
    for n in range(0, l):
        if effR[n] > p:
            U1 = lommel(1, p, effR[n])
            U2 = lommel(2, p, effR[n])
            res += (1 + (U2 ** 2) + (U1 ** 2) + 2*(U2*coslist[n] - U1*sinlist[n]))
        elif effR[n] == p:
            res += (0.25 * ((jv(0, pi*p*p)**2) + 2*cos(pi*p*p)*jv(0, pi*p*p) + 1))
        else:
            res += ((lommel(0, effR[n], p)**2) + (lommel(1, effR[n], p) ** 2))
    return res / l


def fresnel(x, lam, D):

    """ Converts value to fresnel scale units """
    return x / (lam*D/2.)**(1/2.)


def generateKernel(lam, objectR, b, D, starR):

    """ Calculates the light curve at a given wavelength """

    p = fresnel(objectR, lam, D)  # converting KBO radius to Fresnel Scale
    s = fresnel(starR, lam, D)  # converting effective star radius to Fresnel Scale
    b = fresnel(b, lam, D)
    r = 10000.  # distance between line of sight and centre of the disk in m
    z = [diffractionCalc(j, p, s, lam, D, b) for j in np.arange(-r, r, 10)]
    return z


def defineParam(startLam, endLam, objectR, b, D, angDi):

    """ Simulates light curve for given parameters """

    # startLam: start of wavelength range, m
    # endLam: end of wavelength range, m
    # objectR: radius of KBO, m
    # b: impact parameter, m
    # D: distance from KBO to observer, in AU
    # angDi: angular diameter of star, mas
    # Y: light profile during diffraction event

    D *= 1.496e+11 # converting to metres
    starR = effStarRad(angDi, D)
    n = 20
    if endLam == startLam:
        Y = generateKernel(startLam, objectR, b, D, starR)
    else:
        step = (endLam-startLam) / n
        Y = np.array([generateKernel(lam, objectR, b, D, starR) for lam in np.arange(startLam, endLam, step)])
        Y = np.sum(Y, axis=0)
        Y /= n
    return Y


def effStarRad(angDi, D):

    """ Determines projected star radius at KBO distance """

    angDi /= 206265000.  # convert to radians
    return D * tan(angDi / 2.)


def vT(a, vE):

    """ Calculates transverse velocity of KBO """

    # a is distance to KBO, in AU
    # vE is Earth's orbital speed, in m/s
    # returns vT, transverse KBO velocity, in m/s

    return vE * ( 1 - (1./a)**(1/2.))


def integrateCurve(exposure, curve, totTime, shiftAdj):

    """ Reduces resolution of simulated light curve to match what would be observed for a given exposure time """

    timePerFrame = totTime / len(curve)
    numFrames = roundOdd(exposure/timePerFrame)
    if shiftAdj < 0:
        shiftAdj += 1
    shift = ((len(curve) / 2)% numFrames) - (numFrames-1)/2
    while shift < 0:
        shift += numFrames
    shift += int(numFrames*shiftAdj)
    for index in np.arange((numFrames-1)/2 + shift, len(curve)-(numFrames-1)/2, numFrames):
        indices = range(int(index - (numFrames-1)/2), int(index+1+(numFrames-1)/2))
        av = np.average(curve[indices])
        curve[indices] = av
    last = indices[-1]+1  # bins leftover if light curve length isn't divisible by exposure time
    curve[last:] = np.average(curve[last:])
    shift = int(shift)
    curve[:shift] = np.average(curve[:shift])
    return curve, numFrames


def roundOdd(x):

    """ Rounds x to the nearest odd integer """

    x = ceil(x)
    if x % 2 == 0:
        return int(x-1)
    return int(x)


def genCurve(exposure, startLam, endLam, objectRad, impact, dist, angDi, shiftAdj):

    """ Convert diffraction pattern to time series """

    velT = vT(dist, 29800)
    curve = defineParam(startLam, endLam, objectRad, impact, dist, angDi)
    n = len(curve)*10./velT
    curve, num = integrateCurve(exposure, curve, n, shiftAdj)
    return curve[::num]


class OpenKernelUI(QDialog):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.parent = parent
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setModal(True)

        self.ui.pushButton_generate_kernels.clicked.connect(self.generate)
        self.ui.pushButton_reset.clicked.connect(self.reset)

        

    def generate(self):

        #Distance

        distance = self.ui.distance_edit.toPlainText()

        if distance == "":
            distance = 40
        else:
            distance = float(distance)

        #Minimum Lumination

        startLam = self.ui.lumination_edit_min.toPlainText()

        if startLam == "":
            startLam = 4e-7
        else:
            startLam = float(startLam) * 1e-9

        #Maximum Lumination

        endLam = self.ui.lumination_edit_max.toPlainText()

        if endLam == "":
            endLam = 7e-7
        else:
            endLam = float(endLam) * 1e-9

        #Sampling Frequency

        if self.ui.radioButton_freq_5.isChecked():
            samplingFrequency = 5

        elif self.ui.radioButton_freq_17.isChecked():
            samplingFrequency = 17
        
        elif self.ui.radioButton_freq_40.isChecked():
            samplingFrequency = 40

        edit_frequency = self.ui.edit_frequency.toPlainText()
        if edit_frequency != "":
            edit_frequency = float(edit_frequency)
            if edit_frequency < 5 or edit_frequency > 80:
                msg = QMessageBox()
                msg.setWindowTitle("Error")
                msg.setText("Enter Sampling Frequency value between 5 and 80!")

                x = msg.exec_()
            else:
                samplingFrequency = edit_frequency

        exposureTime = 1/samplingFrequency

        #Stellar Diameter

        stellarDiameters = []

        if self.ui.checkBox_stellar_1.isChecked():
            stellarDiameters.append(0.01)
        
        if self.ui.checkBox_stellar_3.isChecked():
            stellarDiameters.append(0.03)

        if self.ui.checkBox_stellar_8.isChecked():
            stellarDiameters.append(0.08)

        edit_stellar_diameter = self.ui.edit_stellar_diameter.toPlainText()
        if edit_stellar_diameter != "":
            edit_stellar_diameter = float(edit_stellar_diameter)
            if edit_stellar_diameter < 0.001 or edit_stellar_diameter > 1:
                msg = QMessageBox()
                msg.setWindowTitle("Error")
                msg.setText("Enter Stellar Diameter value between 0.001 and 1!")

                x = msg.exec_()
            else:
                stellarDiameters.append(edit_stellar_diameter)

        if not self.ui.checkBox_stellar_1.isChecked() and not self.ui.checkBox_stellar_3.isChecked() and not self.ui.checkBox_stellar_8.isChecked() and edit_stellar_diameter == "":
            msg = QMessageBox()
            msg.setWindowTitle("Error")
            msg.setText("Please check atleast one box or enter a value for Stellar Diameter!")

            x = msg.exec_()
            return

        #Object Radius

        objectRadiuses = []

        if self.ui.checkBox_object_1500.isChecked():
            objectRadiuses.append(750)

        if self.ui.checkBox_object_2750.isChecked():
            objectRadiuses.append(1375)

        if self.ui.checkBox_object_5000.isChecked():
            objectRadiuses.append(2500)

        edit_object_diameter = self.ui.edit_object_diameter.toPlainText()
        if edit_object_diameter != "":
            edit_object_diameter = float(edit_object_diameter)
            if edit_object_diameter < 10 or edit_object_diameter > 10000:
                msg = QMessageBox()
                msg.setWindowTitle("Error")
                msg.setText("Enter Object Diameter value between 10 and 10,000!")

                x = msg.exec_()
            else:
                objectRadiuses.append(edit_object_diameter/2)

        if not self.ui.checkBox_object_1500.isChecked() and not self.ui.checkBox_object_2750.isChecked() and not self.ui.checkBox_object_5000.isChecked() and edit_object_diameter == "":
            msg = QMessageBox()
            msg.setWindowTitle("Error")
            msg.setText("Please check atleast one box or enter a value for Object Diameter!")

            x = msg.exec_()
            return

        #Impact Factor

        impactCounter = 0  #To count the number of types of impact factors selected
 
        if self.ui.checkBox_impact_0.isChecked():
            impact_0 = True
            impactCounter += 1
        else:
            impact_0 = False

        if self.ui.checkBox_impact_half_radius.isChecked():
            impact_half_radius = True
            impactCounter += 1
        else:
            impact_half_radius = False

        if self.ui.checkBox_impact_radius.isChecked():
            impact_radius = True
            impactCounter += 1
        else:
            impact_radius = False

        edit_impact_factor = self.ui.edit_impact_factor.toPlainText()
        if edit_impact_factor != "":
            edit_impact_factor = float(edit_impact_factor)
            if edit_impact_factor < 0 or edit_impact_factor > 1:
                msg = QMessageBox()
                msg.setWindowTitle("Error")
                msg.setText("Enter Impact Factor value between 0 and 1!")

                x = msg.exec_()
            else:
                impact_edit = True
                impactCounter += 1
        else:
            impact_edit = False

        if not self.ui.checkBox_impact_0.isChecked() and not self.ui.checkBox_impact_half_radius.isChecked() and not self.ui.checkBox_impact_radius.isChecked() and edit_impact_factor == "":
            msg = QMessageBox()
            msg.setWindowTitle("Error")
            msg.setText("Please check atleast one box or enter a value for Impact Factor!")

            x = msg.exec_()
            return

        #Shift Adjustment

        shiftAdjustments = []

        if self.ui.checkBox_shift_negative_25.isChecked():
            shiftAdjustments.append(-0.25)

        if self.ui.checkBox_shift_0.isChecked():
            shiftAdjustments.append(0)

        if self.ui.checkBox_shift_25.isChecked():
            shiftAdjustments.append(0.25)

        if self.ui.checkBox_shift_5.isChecked():
            #shiftAdjustments.append(5)     
            shiftAdjustments.append(0.5)      #Edit: 5 to 0.5 - RAB 031722

        edit_shift_adjustment = self.ui.edit_shift_adjustment.toPlainText()
        if edit_shift_adjustment != "":
            edit_shift_adjustment = float(edit_shift_adjustment)
            if edit_shift_adjustment < -0.5 or edit_shift_adjustment > 0.5:
                msg = QMessageBox()
                msg.setWindowTitle("Error")
                msg.setText("Enter Shift Adjustment value between -0.5 and 0.5!")

                x = msg.exec_()
            else:
                shiftAdjustments.append(edit_shift_adjustment)

        if not self.ui.checkBox_shift_negative_25.isChecked() and not self.ui.checkBox_shift_0.isChecked() and not self.ui.checkBox_shift_25.isChecked() and not self.ui.checkBox_shift_5.isChecked() and edit_shift_adjustment == "":
            msg = QMessageBox()
            msg.setWindowTitle("Error")
            msg.setText("Please check atleast one box or enter a value for Shift Adjustment!")

            x = msg.exec_()
            return

        
        #Generating Kernels:

        self.progressBar = self.ui.progressBar

        totalKernels = impactCounter * len(shiftAdjustments) * len(objectRadiuses) * len(stellarDiameters)

        self.progressBar.setMaximum(totalKernels)

        kernels = []

        kernel_params = []      #list to hold kernel parameters - RAB 031522
        
        progressCounter = 0
        
        if impact_0:
            for objectRadius in objectRadiuses:
                for stellarDiameter in stellarDiameters:
                    for shiftAdjustment in shiftAdjustments:
                        
                        kernel_params.append((progressCounter, samplingFrequency, objectRadius, stellarDiameter, 0, shiftAdjustment))   #add kernel parameters to array - RAB 031522
                        
                        kernel = genCurve(exposureTime, startLam, endLam, objectRadius, 0, distance, stellarDiameter, shiftAdjustment)
                        kernels.append(kernel)
                        plt.figure()
                        plt.xlabel('time (s)')
                        plt.ylabel('fractional intensity')
                        graphName = str(progressCounter) + '_' + str(samplingFrequency) + "Hz_" + str(objectRadius) + "m_" + str(stellarDiameter) + "mas_0m_" + str(shiftAdjustment) + "frames.png"  #Added kernel index to plot title - RAB 031522
                        plt.title(graphName)
                        plt.plot(kernel)
                        path = "kernel_images/" + graphName
                        plt.savefig(path)
                        progressCounter = progressCounter + 1
                        self.ui.progress_Label.setText(str(progressCounter) + "/" + str(totalKernels))
                        self.progressBar.setValue(self.progressBar.value() + 1)


        if impact_half_radius:
            for objectRadius in objectRadiuses:
                for stellarDiameter in stellarDiameters:
                    for shiftAdjustment in shiftAdjustments:
                        
                        kernel_params.append((progressCounter, samplingFrequency, objectRadius, stellarDiameter, objectRadius/2, shiftAdjustment))   #add kernel parameters to array - RAB 031522
                        
                        kernel = genCurve(exposureTime, startLam, endLam, objectRadius, objectRadius/2, distance, stellarDiameter, shiftAdjustment)
                        kernels.append(kernel)
                        plt.figure()
                        plt.xlabel('time (s)')
                        plt.ylabel('fractional intensity')
                        graphName = str(progressCounter) + '_' + str(samplingFrequency) + "Hz_" + str(objectRadius) + "m_" + str(stellarDiameter) + "mas_" + str(objectRadius/2) + "m_" + str(shiftAdjustment) + "frames.png"    #added kernel index - RAB 031522
                        plt.title(graphName)
                        plt.plot(kernel)
                        path = "kernel_images/" + graphName
                        plt.savefig(path)
                        progressCounter = progressCounter + 1
                        self.ui.progress_Label.setText(str(progressCounter) + "/" + str(totalKernels))
                        self.progressBar.setValue(self.progressBar.value() + 1)
                        
                        
        if impact_radius:
            for objectRadius in objectRadiuses:
                for stellarDiameter in stellarDiameters:
                    for shiftAdjustment in shiftAdjustments:
                        
                        kernel_params.append((progressCounter, samplingFrequency, objectRadius, stellarDiameter, objectRadius, shiftAdjustment))   #add kernel parameters to array - RAB 031522
                        
                        kernel = genCurve(exposureTime, startLam, endLam, objectRadius, objectRadius, distance, stellarDiameter, shiftAdjustment)
                        kernels.append(kernel)
                        plt.figure()
                        plt.xlabel('time (s)')
                        plt.ylabel('fractional intensity')
                        graphName = str(progressCounter) + '_' + str(samplingFrequency) + "Hz_" + str(objectRadius) + "m_" + str(stellarDiameter) + "mas_" + str(objectRadius) + "m_" + str(shiftAdjustment) + "frames.png"         #added kernel index = RAB 031522
                        plt.title(graphName)
                        plt.plot(kernel)
                        path = "kernel_images/" + graphName
                        plt.savefig(path)
                        progressCounter = progressCounter + 1
                        self.ui.progress_Label.setText(str(progressCounter) + "/" + str(totalKernels))
                        self.progressBar.setValue(self.progressBar.value() + 1)
                        
        if impact_edit:
            for objectRadius in objectRadiuses:
                for stellarDiameter in stellarDiameters:
                    for shiftAdjustment in shiftAdjustments:
                        
                        kernel_params.append((progressCounter, samplingFrequency, objectRadius, stellarDiameter, objectRadius + edit_impact_factor, shiftAdjustment))   #add kernel parameters to array - RAB 031522
                        
                        kernel = genCurve(exposureTime, startLam, endLam, objectRadius, objectRadius * edit_impact_factor, distance, stellarDiameter, shiftAdjustment)
                        kernels.append(kernel)
                        plt.figure()
                        plt.xlabel('time (s)')
                        plt.ylabel('fractional intensity')
                        graphName = str(progressCounter) + '_' + str(samplingFrequency) + "Hz_" + str(objectRadius) + "m_" + str(stellarDiameter) + "mas_" + str(objectRadius * edit_impact_factor) + "m_" + str(shiftAdjustment) + "frames.png"     #added kernel index - RAB 031522
                        plt.title(graphName)
                        plt.plot(kernel)
                        path = "kernel_images/" + graphName
                        plt.savefig(path)
                        progressCounter = progressCounter + 1
                        self.ui.progress_Label.setText(str(progressCounter) + "/" + str(totalKernels))
                        self.progressBar.setValue(self.progressBar.value() + 1)


        fileName = self.ui.file_name_edit.toPlainText()
        if fileName == "":
            fileName = "Kernels.txt"
        elif ".txt" not in fileName:
            fileName = fileName + ".txt"
        
        np.savetxt(fileName, kernels)#('kernelstest.txt', kernels)
        
        #Added in .txt file to keep track of kernel parameters - RAB 031522
        param_filename = 'params_' + fileName
        
        with open(param_filename, "w") as filehandle:
        
            filehandle.write('#KernelIndex    samplingFreq    ObjectRadius    StellarDiameter    ImpactParameter   ShiftAdjustment\n')
        
            for line in kernel_params:
            
                filehandle.write('%i %i %f %f %f %f\n' %(line[0], line[1], line[2], line[3], line[4], line[5]))
        #-----------end of RAB addition----------------------------------

    def reset(self):
        self.ui.checkBox_impact_0.setChecked(False)
        self.ui.checkBox_impact_half_radius.setChecked(False)
        self.ui.checkBox_impact_radius.setChecked(False)
        self.ui.checkBox_object_1500.setChecked(False)
        self.ui.checkBox_object_2750.setChecked(False)
        self.ui.checkBox_object_5000.setChecked(False)
        self.ui.checkBox_shift_0.setChecked(False)
        self.ui.checkBox_shift_25.setChecked(False)
        self.ui.checkBox_shift_5.setChecked(False)
        self.ui.checkBox_shift_negative_25.setChecked(False)
        self.ui.checkBox_stellar_1.setChecked(False)
        self.ui.checkBox_stellar_3.setChecked(False)
        self.ui.checkBox_stellar_1.setChecked(False)
        self.ui.lumination_edit_min.setPlainText("")
        self.ui.lumination_edit_max.setPlainText("")
        self.ui.distance_edit.setPlainText("")
        self.ui.file_name_edit.setPlainText("")
        self.ui.radioButton_freq_5.setChecked(True)
        self.ui.progressBar.setValue(0)
        self.ui.progress_Label.setText("")
        self.ui.edit_frequency.setPlainText("")
        self.ui.edit_impact_factor.setPlainText("")
        self.ui.edit_object_diameter.setPlainText("")
        self.ui.edit_shift_adjustment.setPlainText("")
        self.ui.edit_stellar_diameter.setPlainText("")
        




if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = OpenKernelUI()
    win.show()
    sys.exit(app.exec_())


# class modler():

#     def __init__(self):
#         self.generator_ui = OpenChatbotUI(self)

#     #generator_ui.show()


# m = modler()
















# objectRadius = int(input("Enter object diameter(m)[1500, 2750, 5000] = "))//2
# stellarDiameter = float(input("Enter stellar diameter(mas)[0.01, 0.03, 0.08] = "))
# impactParameter = int(input("Enter impact parameter(m)[0, halfObjectRadius, objectRadius] = "))
# shiftAdjustment = float(input("Enter shift adjustment(frame)[0, 0.25, 0.5] = "))
# samplingFrequency = float(input("Enter sampling frequency(Hz)[5,17,40] = "))
# exposureTime = 1/samplingFrequency

# #390nm to 800nm
# #kernels = [genCurve(0.05991, 4e-7, 7e-7, objR, imp, 40, angDi, shiftAdj) for angDi in [0.01, 0.03, 0.08] for objR in [750, 1375, 2500] for shiftAdj in [0, 0.25, 0.5] for imp in [0, objR]]
# kernels = genCurve(exposureTime, 39e-8, 8e-7, objectRadius, impactParameter, 40, stellarDiameter, shiftAdjustment)

# plt.plot(kernels)
# plt.savefig("kernel_images/abc.png")


# np.savetxt('kernel_dip_test.txt', kernels)#('kernelstest.txt', kernels)
