FROM detectron2_images:20191204 
RUN git clone https://github.com/zj463261929/detectron2-restful /detectron2

COPY model.pth /detectron2/restful/model/
COPY RetinaNet.yaml /detectron2/restful/model/

# Set up Python modules
WORKDIR /detectron2
RUN python setup.py build develop

# install vim
#RUN apt-get update
#RUN apt-get install vim
#RUN rm -rf /var/lib/apt/lists/*

# start restful service
WORKDIR /detectron2/restful
CMD python app.py
EXPOSE 8080