# Sound Event Detection Web

This project is a prototype web app for Sound Event Detection.

Implemented using Flask

This project using Python 3.7.6 version

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

* Setup virtual enviroment
  ```shell
  env\Scripts\activate.bat
  ```
* Start Development Server
  ```shell
  python app.py
  ```

By Prayudha Adhitia Libramawan