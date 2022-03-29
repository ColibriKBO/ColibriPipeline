# Kernel Generator GUI
This is a Graphical User Interface for generating Kernels.
The GUI lets the user select various parameters using which the kernels are generated such as:
- Distance(au): Approximate distance of the user from the star (40au). 
- Wavelength Bands(nm): Minimum and maximum limits of the wavelength bands observed.
- Sampling Frequency(Hz): Number of images taken per second.
- Object Diameter(m): Expected diameter of the object to be detected.
- Stellar Diameter(mas): Approximate diameter of the stars observed.
- Impact Factor(units of radius): Maximum part of the star that the detected object will cover when passing in front of the star. 
- Shift Adjustment(frames): Part of the exposure time after which the occultation begins.  
  
Following is the image of the Graphical User Interface:

![Kernel Generator GUI Image](images/KernelGeneratorGUI.png?raw=true "Kernel Generator GUI")

## Cloning this repository
```
git https://github.com/ColibriKBO/KernelGeneratorGUI.git
cd KernelGeneratorGUI
```

## Dependencies
These Dependencies must be installed prior to running the Kernel Generator GUI:

### Automatic installation
Run the instDep.sh file which will install all dependencies and create the folder:
```
sh instdep.sh
```

### OR

### Manual installation
For manually installing the dependencies follow the following steps:

> Make python3 as default using the following command
```
echo "alias python=python3" >> ~/.bash_aliases
source ~/.bash_aliases
```

> Upgrade pip to latest version using the following command:
```
python3 -m pip install --upgrade pip
```

#### Python Library Installations
```
pip3 install PySide2
pip3 install numpy
pip3 install scipy
pip3 install matplotlib
```
#### Creating folder to save graphs of kernels generated
```
mkdir kernel_images
```


## Starting the GUI
To start the Kernel Generator GUI using the follwing command:
```
pip3 fresnelModelerGUI.py
```





