#!/bin/bash

# Check Python version
python_version=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
required_version="3.9"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
    echo "Python version $python_version meets requirement (>= $required_version)"
else
    echo "Error: Python >= $required_version required. Current version: $python_version"
    exit 1
fi

# Create and activate virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment (Windows compatible)
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install required packages
echo "Installing required packages..."
pip install pandas numpy scikit-learn requests python-dateutil matplotlib joblib

# Export FRED API key
export FRED_API_KEY="0a5af412d677d2f55e14ddd03b499eae"

# Verify API key is set
if [ -z "$FRED_API_KEY" ]; then
    echo "Error: FRED_API_KEY is not set or empty"
    exit 1
fi

echo "Setup completed successfully! Virtual environment created and FRED API key exported."
echo "To activate the environment in future sessions, run: source venv/bin/activate (Linux/Mac) or source venv/Scripts/activate (Windows)"