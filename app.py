from flask import Flask, render_template, jsonify
import datetime
import os

app = Flask(__name__)

# Simple HTML template
@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>iPhone Code Viewer</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
                min-height: 100vh;
                padding: 20px;
                color: white;
            }
            
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            }
            
            h1 {
                text-align: center;
                margin-bottom: 20px;
                font-size: 2.5em;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
            }
            
            .info-box {
                background: rgba(255, 255, 255, 0.15);
                padding: 20px;
                border-radius: 15px;
                margin: 15px 0;
                font-size: 1.1em;
            }
            
            .code-block {
                background: rgba(0, 0, 0, 0.3);
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
                font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
                font-size: 0.9em;
                overflow-x: auto;
            }
            
            .status-indicator {
                display: inline-block;
                width: 12px;
                height: 12px;
                background: #4CAF50;
                border-radius: 50%;
                margin-right: 10px;
                animation: pulse 2s infinite;
            }
            
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            
            .button {
                display: inline-block;
                background: white;
                color: #6a11cb;
                padding: 12px 25px;
                border-radius: 50px;
                text-decoration: none;
                font-weight: bold;
                margin: 10px 5px;
                transition: all 0.3s ease;
                border: none;
                font-size: 1em;
                cursor: pointer;
            }
            
            .button:hover {
                transform: translateY(-3px);
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
            }
            
            .terminal {
                background: #1a1a1a;
                color: #00ff00;
                padding: 15px;
                border-radius: 10px;
                font-family: 'Courier New', monospace;
                margin: 20px 0;
                max-height: 200px;
                overflow-y: auto;
            }
            
            .file-list {
                display: grid;
                gap: 10px;
                margin: 20px 0;
            }
            
            .file-item {
                background: rgba(255, 255, 255, 0.1);
                padding: 15px;
                border-radius: 10px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .file-icon {
                font-size: 1.5em;
            }
            
            @media (max-width: 600px) {
                .container {
                    padding: 20px;
                }
                
                h1 {
                    font-size: 2em;
                }
                
                .button {
                    display: block;
                    margin: 10px 0;
                    width: 100%;
                    text-align: center;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📱 iPhone Code Viewer</h1>
            
            <div class="info-box">
                <span class="status-indicator"></span>
                <strong>Status:</strong> Flask app is running!
            </div>
            
            <div class="info-box">
                <h3>🚀 Quick Actions</h3>
                <div style="margin-top: 15px;">
                    <button class="button" onclick="checkStatus()">Check Status</button>
                    <button class="button" onclick="getTime()">Get Server Time</button>
                    <button class="button" onclick="listFiles()">List Files</button>
                </div>
            </div>
            
            <div class="info-box">
                <h3>📊 Server Information</h3>
                <p id="server-info">Click buttons above to load info...</p>
            </div>
            
            <div class="terminal">
                <div id="output">Terminal output will appear here...</div>
            </div>
            
            <div class="info-box">
                <h3>📖 How to Use</h3>
                <ol style="margin-left: 20px; margin-top: 10px;">
                    <li>Edit code in GitHub Codespaces web editor</li>
                    <li>Save your changes</li>
                    <li>This page auto-refreshes every 10 seconds</li>
                    <li>Check the terminal output below</li>
                </ol>
            </div>
            
            <div class="info-box">
                <h3>🔗 Your Ports</h3>
                <p>Main Port: <strong id="port-number">Loading...</strong></p>
                <p>Codespaces URL: <code id="codespace-url">Loading...</code></p>
                <button class="button" onclick="openInSafari()">Open in Safari</button>
            </div>
        </div>
        
        <script>
            // Get port from URL
            const port = window.location.port || '5000';
            document.getElementById('port-number').textContent = port;
            document.getElementById('codespace-url').textContent = window.location.hostname;
            
            // Auto-refresh every 10 seconds
            setTimeout(() => {
                window.location.reload();
            }, 10000);
            
            async function checkStatus() {
                const response = await fetch('/api/status');
                const data = await response.json();
                document.getElementById('server-info').innerHTML = `
                    <strong>Status:</strong> ${data.status}<br>
                    <strong>Time:</strong> ${data.time}<br>
                    <strong>Port:</strong> ${data.port}
                `;
                addOutput('✓ Status checked at ' + new Date().toLocaleTimeString());
            }
            
            async function getTime() {
                const response = await fetch('/api/time');
                const data = await response.json();
                document.getElementById('server-info').innerHTML = `
                    <strong>Server Time:</strong> ${data.time}<br>
                    <strong>Timezone:</strong> ${data.timezone}
                `;
                addOutput('✓ Time requested at ' + new Date().toLocaleTimeString());
            }
            
            async function listFiles() {
                const response = await fetch('/api/files');
                const data = await response.json();
                let fileList = '<strong>Files in directory:</strong><br>';
                data.files.forEach(file => {
                    fileList += `📄 ${file}<br>`;
                });
                document.getElementById('server-info').innerHTML = fileList;
                addOutput('✓ File list requested at ' + new Date().toLocaleTimeString());
            }
            
            function addOutput(message) {
                const output = document.getElementById('output');
                const time = new Date().toLocaleTimeString();
                output.innerHTML = `<div>[${time}] ${message}</div>` + output.innerHTML;
            }
            
            function openInSafari() {
                const url = window.location.href;
                window.open(url, '_blank');
                addOutput('✓ Opened in new tab');
            }
            
            // Initial status check
            checkStatus();
        </script>
    </body>
    </html>
    '''

@app.route('/api/status')
def api_status():
    return jsonify({
        'status': 'running',
        'service': 'Flask App on iPhone',
        'time': datetime.datetime.now().strftime('%H:%M:%S'),
        'port': os.environ.get('PORT', 5000)
    })

@app.route('/api/time')
def api_time():
    return jsonify({
        'time': datetime.datetime.now().isoformat(),
        'timezone': 'UTC',
        'timestamp': datetime.datetime.now().timestamp()
    })

@app.route('/api/files')
def api_files():
    files = [f for f in os.listdir('.') if f.endswith(('.py', '.txt', '.json', '.html', '.css', '.js'))]
    return jsonify({'files': files})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("=" * 50)
    print("🚀 FLASK APP STARTED")
    print(f"📱 Access on iPhone Safari:")
    print(f"   http://localhost:{port}")
    print("=" * 50)
    print("💡 TIPS:")
    print("1. Edit app.py and save changes")
    print("2. Refresh Safari to see updates")
    print("3. Check terminal for logs")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=port, debug=True)
