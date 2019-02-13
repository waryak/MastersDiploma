# MastersDiploma
Authors: Vladislav Ladenkov (waryak2012@mail.ru)   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Alexander Vorontsov (запили сюда свое мыло)

This repository contains code and papers to our master's degree diploma. 
The diploma's subject is *"Finfing figures of technical analysis in cryptocurrency exchange time serieses"*.

The code base here is mainly about distributed computing of Wishart algorithm using message queueing. 

>BRIEF EXPLANATION ON PROJECT 
   
>PICTURE HERE
### Project Structure Overview
```
├── configs      # Configs for worker role, producer role and evaluater role
├── dockerfilers # Dockerfiles for worker, producer and evaluater
├── docs         # Documentation, tutorials, etc.
├── notebooks    # Research and examples
├── paperrs      # Relevent academic papers
└── src
    ├── algo      # Contains Wishart algo and math tools to work with chaotic time-serieses
    ├── datamart  # Database uploads/downloads, models storing/preprocessing 
    ├── evaluater # Predicter, evaluater
    └── network   # Rabbit MQ + celery
```
### Documentation:
1. [algo] 
2. [datamart]
3. [evaluater](docs/images_build.md)
4. [network]

### Quick start

Every part of the project is dockerized, 
so everything should be easy as soon as you set up your machines to run docker correctly.     
Schematically, you should do this:  
**1.** Install docker(18.09.0 was used) and docker-compose(1.23.1 was used) on all your machines
```bash
# Installing docker-ce
sudo apt-get update
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo apt-get install docker-ce
# Adding a new user to the «docker» group
sudo usermod -aG docker $USER
# Tmux installation
sudo apt-get install tmux
```
**2.** Pull project from github (Later, it would be upgraded to CI)
```bash
git clone https://github.com/waryak/MastersDiploma.git
cd MastersDiploma
```
**3.** Build services for worker and producer roles:
```bash
# For producer
docker-compose build wishart-producer
# For worker
docker-compose build wishart-worker
```
**4.** Launch services for worker and producer roles:
```bash
# For producer
docker-compose up wishart-producer
# For worker
docker-compose up wishart-worker
```




