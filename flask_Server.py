from flask import Flask, render_template, request, jsonify
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/mode', methods=['POST'])
def set_mode():
    mode = request.json.get('mode')  # 🔥 JSON input from JS
    if mode:
        subprocess.Popen(["python", "main.py", mode])
        return jsonify({"status": "success", "message": f"{mode} activated"})
    return jsonify({"status": "error", "message": "Invalid mode"}), 400

if __name__ == '__main__':
    app.run(debug=True)