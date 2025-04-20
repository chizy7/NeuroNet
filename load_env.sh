#!/bin/bash
# load_env.sh

if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "Environment variables loaded"
else
    echo "Error: .env file not found"
    echo "Please copy .env.template to .env and fill in your values"
    exit 1
fi