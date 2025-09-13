#!/usr/bin/env bash

echo "Creating virtual environment"
python3 -m venv vibe-env
echo "Activating virtual environment"

source $PWD/vibe-env/bin/activate

$PWD/vibe-env/bin/pip install numpy torch==1.4.0 torchvision==0.5.0
$PWD/vibe-env/bin/pip install git+https://github.com/giacaglia/pytube.git --upgrade
$PWD/vibe-env/bin/pip install -r requirements.txt
