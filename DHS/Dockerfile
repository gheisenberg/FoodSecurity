# Start with a base image (choose an appropriate one, e.g., Ubuntu, Python)
FROM tensorflow/tensorflow:2.16.1-gpu-jupyter

LABEL sven woehrle <sven.woehrle@th-koeln.de>

# Update package manager and install necessary tools
RUN apt-get update && apt-get install -y \
    sudo \
    build-essential \
    && apt-get clean

ARG HOST_UID=1000
ARG HOST_GID=1000
RUN groupadd -g ${HOST_GID} mygroup && \
    useradd -m -u ${HOST_UID} -g mygroup -s /bin/bash myuser && \
    echo "myuser:password" | chpasswd

# Switch to the new user
USER myuser
WORKDIR /home/myuser

RUN pip3 -q install pip --upgrade
RUN pip install --upgrade pip setuptools
#copy requirements and install them
COPY requirements_wo_versions.txt .
RUN pip install -r requirements_wo_versions.txt

# Set environment variable to add .local/bin to PATH
ENV PATH="/home/myuser/.local/bin:$PATH"

USER root
# Copy the entrypoint script into the container
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

USER myuser

# Start Jupyter Lab automatically
CMD ["bash"]