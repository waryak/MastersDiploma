#### How to build an image of the worker
To build a worker you need to execute.  
`docker-compose build wishart-worker`  
It is essential to have a config file `./configs/config.yml` in the workspace of directory

#### How to navigate through volume data 
`docker run --rm -it -v mastersdiploma_wishart-data:/mnt/ ubuntu:16.04 /bin/bash`