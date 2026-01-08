# Python Git Linux Project

## Overview
This project is a web-based financial dashboard developed in Python using the Streamlit framework. It was created as part of the "Python for Finance" and "Linux/Git" coursework within the Master 1 Financial Engineering curriculum at ESILV.

The application serves as a comprehensive tool for analyzing financial assets, combining interactive visualizations with backend automation.

## Features

### 1. Single Asset Dashboard
A specialized module designed for the in-depth analysis of individual assets.
* **Persistent Sidebar Logic:** Implements a specific UX pattern where essential metrics and control parameters are locked into the sidebar when this module is active, ensuring key data remains visible during analysis.
* **Performance Metrics:** Real-time calculation of returns, volatility, and historical performance.

### 2. Multi-Asset Dashboard
* [This section is currently under development. Add details about correlation matrices, portfolio optimization, or comparative analysis here.]

### 3. Automation & System Integration
* **Cron Jobs:** The project includes backend scripts designed to run on a Linux server to automate data fetching and daily reporting.
* **Version Control:** Structured to demonstrate best practices in Git branching and merging workflows.

## Technical Stack

* **Language:** Python 3.10+
* **Web Framework:** Streamlit
* **Data Processing:** Pandas, NumPy
* **Visualization:** Plotly / Matplotlib
* **Environment:** Linux (Ubuntu/Debian) for automation

## Installation

To set up the project locally, follow these steps:

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/YourUsername/PythonGitLinuxProject.git](https://github.com/YourUsername/PythonGitLinuxProject.git)
    cd PythonGitLinuxProject
    ```

2.  **Create a virtual environment**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # Mac/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To launch the dashboard interface:

```bash
streamlit run app.py
