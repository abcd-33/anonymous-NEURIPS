#!/bin/bash

set -e  
echo -e "\n--- Starting Data Pipeline ---\n"

echo -e "\n[1/2] Preprocessing data..."
python preprocess_data.py
echo "Preprocessing complete."

echo -e "\n[2/2] Training model..."
python train.py
echo "Training complete."

echo -e "\n--- Pipeline finished successfully! ---\n"