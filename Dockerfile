FROM python:3.10.10

# Creating Application Source Code Directory
RUN mkdir -p /usr/src/app

# Setting Home Directory for containers
WORKDIR /usr/src/app

# Installing python dependencies
COPY ./requirements.txt /usr/src/app/

RUN apt-get update && apt-get install --yes libgdal-dev

# RUN pip install GDAL

# # Install GDAL dependencies
# RUN apt-get install -y libgdal-dev g++ --no-install-recommends && \
#     apt-get clean -y

# # Update C env vars so compiler can find gdal
# ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
# ENV C_INCLUDE_PATH=/usr/include/gdal

RUN pip install --no-cache-dir -r requirements.txt

# Copying src code to Container
COPY . /usr/src/app

# Application Environment variables
#ENV APP_ENV development
ENV PORT 8501

# Exposing Ports
EXPOSE $PORT

# Setting Persistent data
# VOLUME ["/app-data"]

# Running Python Application
# CMD ["python", "./main.py"]

ENTRYPOINT ["streamlit", "run"]

CMD ["main.py"]