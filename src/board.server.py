from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import random

import gameplay
import numpy as np

hostName = "localhost"
serverPort = 8000
#TO DO: fix below
htmlFilePath = "src/gomoku-board.html"
playersDict = {1: 'p1',2: 'p2', -1: 'p2'} # This should not have -1 , but player 2 start bug


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

            counter = 0
            board = np.zeros((env.dim, env.dim))
            indexCount = 0
            # while indexCount < len(data):
            #     v = data[indexCount]
            for v in data['board']:
                if v.isnumeric():
                    pot = int(counter/env.dim)
                    oth = counter % env.dim
                    board[pot][oth] = int(v)
                    counter +=1

            print(board)
            print("****3***")
            curplayer = playersDict[data['player']]

            env.importBoard(board)
            state, coor, gameIsDone = gameplay.make_inference(env, curplayer, model, device)
            print(coor)
            random_row, random_col = coor
            # Add your custom handling for the POST request data here
            #I will eventually add proper code to handle different routes it's just not 
            # necessary atm since the only thing making POST requests is the front-end 
            #and its only ever to get the next move

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            # random_row = random.randint(0, 14)
            # random_col = random.randint(0,14)
            coor = str(int(random_row))+"-"+str(int(random_col))
            response_data = {'move': coor}
            self.wfile.write(json.dumps(response_data).encode('utf-8'))
            print("ran")

        except json.JSONDecodeError:
            self.send_response(400)  # Bad Request
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Invalid JSON data')



if __name__ == "__main__":        
    model, device, env = gameplay.get_Model_Device_Environment()

    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")