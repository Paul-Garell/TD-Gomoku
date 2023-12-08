from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import random

hostName = "localhost"
serverPort = 8000
htmlFilePath = "/Users/lcali/Desktop/ai-prac/TD-Gomoku/gomoku-gui/gomoku-board.html"

class MyServer(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
          with open(htmlFilePath, "rb") as file:
              content = file.read()
          self.send_response(200)
          self.send_header("Content-type", "text/html")
          self.end_headers()
          self.wfile.write(content)
        except FileNotFoundError:
            self.send_response(404)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(bytes("<html><head><title>File Not Found</title></head>", "utf-8"))
            self.wfile.write(bytes("<body><p>File not found.</p></body></html>", "utf-8"))
        #print("get executed")

    def do_POST(self):
        #print("POSTING")
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        if self.path == "/receive_state":
            print("wants next move")

        try:
            data = json.loads(post_data.decode('utf-8'))
            #print('Received POST data:', data)

            # Add your custom handling for the POST request data here

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            random_row = random.randint(0, 14)
            random_col = random.randint(0,14)
            coor = str(random_row)+"-"+str(random_col)
            response_data = {'move': coor}
            self.wfile.write(json.dumps(response_data).encode('utf-8'))

        except json.JSONDecodeError:
            self.send_response(400)  # Bad Request
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Invalid JSON data')



if __name__ == "__main__":        
    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")