#!/bin/bash
# Adjust permissions for the SSL files
chmod 644 /etc/ssl/mycert.pem
chmod 644 /etc/ssl/mykey.key

# Execute the passed CMD arguments
exec "$@"
