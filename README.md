# Sound Event Detection Web

This project is a prototype web app for Sound Event Detection using Convolutional Gated Recurrent Unit Method.

Implemented using Flask and Python 3.7.9 version

## Directory Structure
```
Flask
├── env                     // a Python virtual environment where Flask and other dependencies are installed.
├── sed_vis                 // an external library that used for visualize the prediction
├── static                  // a directory containing saved files.
│   └── audio                   // the app will save the audio files in this directory
│   └── plt                     // the app will save the plot images in this directory
│   └── results                 // the app will save the results of prediction in this directory
└── templates               // presentation
```

## How to Run

* Setup Virtual Enviroment
  ```shell
  pip install virtualenv

  virtualenv env

  env\\Scripts\\activate.bat
  ```
* Setup Depedencies
  ```shell
  pip install -r requirements.txt
  ```
* Start Development Server
  ```shell
  python app.py
  ```