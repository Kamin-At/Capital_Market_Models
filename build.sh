#!/usr/bin/env bash
echo "Checking all the requirements"
pip install requirements.txt
echo "Checking test cases"
pytest