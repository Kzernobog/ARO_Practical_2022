# ARO Practical 2022

![Git](https://github.com/favicon.ico)

Hello and welcome to the ARO Practical 2022! This repository contains the code from which you will develop your own solutions to each of the tasks described in the Practical Guidebook. To get started, clone the repository to your local machine:

```
$ git clone https://github.com/avro22/ARO_Practical_2022.git
$ cd ./ARO_Practical_2022
```

<p align="center">
    <img height="300" src="/Nextage_robot.png">
</p>

## Setting up the environment:
From here, you should be under ./ARO_Practical_2022 directory.

### Option 1: Setup with conda
Make sure you have either miniconda or anaconda installed. If not, you can follow the instructions on the [website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to get conda. On DICE, conda is already installed and up-to-date, so you can ignore the following command. To make sure your conda is up-to-date, run:
```
$ conda update conda
```

Conda automatically installs everything you need and create an isolated envirnoment. Run following code in your terminal to create a conda environment called ARO_Practical. 
```
$ conda env create --file environment.yml
$ # If you wish to name it differently, change <YOURENVNAME> with following command: 
$ conda env create --name <YOURENVNAME> --file environment.yml
```

When it is all finished, run following command to start your conda environment and you are good to go!
```
$ conda activate ARO_Practical
```

### Option 2: Setup up with pip
If you are not a fan of conda, you can get everything working with pip, too! But, you will have to manually install some packages yourself which is not so hard.
First, let’s install python 3.6 and pip3. Run following commands:
```
$ sudo apt update && sudo apt upgrade
$ sudo apt install python3.6
$ wget https://bootstrap.pypa.io/get-pip.py -P /tmp 
$ python3 /tmp/get-pip.py
```
To verify your python version is 3.6.x and pip version is 20.x, run:
```
$ python3 --version
$ python3 -m pip --version
```

Now, run this command to install packages:
```
$ python3 -m pip install -r requirement.txt
```

Congratulations, you are now ready to start!

## Hello world Pybullet!
To test the installation, you can run some hello-world programs:
```
(base)$ conda activate ARO_Practical
(ARO_Practical)$ python3 hello_world.py
(ARO_Practical)$ python3 hello_Nextage.py
```


### Dependency list
| Package    |   Version   |
|------------|-------------|
| Pybullet   | 2\.8\.7     |
| Scipy      | >= 1\.5\.2  |
| Numpy      | >= 1\.19\.1 |
| Matplotlib | >= 3\.3\.1  |
| PyYAML     | >= 5\.3\.1  |


## Trouble shooting
Please see [Trouble shooting](https://github.com/AVRO22/ARO_Practical_2022/blob/main/trouble_shooting.md)
