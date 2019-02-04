
Launching container with worker  
`docker run --rm -v <volume_name>:/var/data -it <image_name>`
Launching container with producer 
`docker run --rm -v <volume_name>:/var/data -it --entrypoint /bin/bash <image_name>`
