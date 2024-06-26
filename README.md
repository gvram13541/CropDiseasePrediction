# Leaf Disease Prediction

This project aims to predict leaf diseases using a Convolutional Neural Network (CNN). The frontend is built with React, and the backend is implemented using Flask. Additionally, the project utilizes the YouTube API and Beautiful Soup to fetch related articles and videos.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [License](#license)

## Introduction
This project helps in identifying diseases in plant leaves by analyzing images. It leverages deep learning techniques to accurately predict the disease. To provide additional resources, it fetches related articles and videos using Beautiful Soup and the YouTube API.

## Features
- Upload leaf images for disease prediction camera and file supportable
- CNN-based image classification for accurate disease detection
- React-based user-friendly interface
- Flask backend for handling API requests and machine learning model integration
- YouTube API integration for related video suggestions
- Web scraping with BeautifulSoup to fetch relevant articles

## Technologies Used
- **Frontend:** React
- **Backend:** Flask
- **Machine Learning:** Convolutional Neural Network (CNN)
- **Web Scraping:** Beautiful Soup
- **API Integration:** YouTube API

## Setup and Installation

### Prerequisites
- Python 3.x
- Node.js and npm
- Flask
- React
- API
- Virtual environment

### Setup
1. Clone the repository:
    ```bash
    git clone https://github.com/gvram13541/CropDiseasePrediction.git
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the script:
    ```bash
    ./start-app.bat  # on Windows (if using linux create .sh file and run it)
    ```

## Usage
1. Open your web browser and navigate to `http://localhost:3000`.
2. Upload an image of a leaf to predict the disease.
3. View the prediction results along with related articles and YouTube videos.
