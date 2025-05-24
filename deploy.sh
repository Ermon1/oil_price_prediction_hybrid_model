#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Create models directory
mkdir -p /opt/render/project/src/models

# Copy models from local models directory
if [ -d "models" ]; then
    echo "Copying models..."
    cp -r models/* /opt/render/project/src/models/
    echo "Models copied successfully"
else
    echo "Error: models directory not found"
    exit 1
fi

# Start the application
exec python main.py 