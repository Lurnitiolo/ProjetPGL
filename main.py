import flask
import os

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def home():
    return "main page"

@app.route("/api/daily_report", methods=['GET'])
def daily_report():
    # Test the existance of "report/daily_report_latest.csv" and return its content
    
    file_path = "Arthur/reports/daily_report_latest.csv"
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return file.read()
    else:
        return "Report not found."

app.run()