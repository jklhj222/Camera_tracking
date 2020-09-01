#!/usr/bin/env python3

import socket

class SocketSanta():
    def __init__(self, HOST, PORTS):
        # HOST: server host ip
        # PORT: (receive_port, send_port)
        self.host = HOST
        self.recv_port = PORTS[0]
        self.send_port = PORTS[1]

#        self.s_receive, self.s_send = self.connect()

#        self.conn_receive = self.conn_send = None
#        while self.conn_receive is None and self.conn_send is None:
#            self.conn_receive, self.conn_send = \
#                self.accept(self.s_receive, self.s_send)


    def connect(self, host, recv_port, send_port):
        s_receive = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        s_receive.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s_receive.bind((host, recv_port))
        s_receive.listen(5)

        s_send.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s_send.bind((host, send_port))
        s_send.listen(5)

        print('Server start at: {}:{} for receiving'.format(host, 
                                                            recv_port))

        print('Server start at: {}:{} for sending'.format(host,
                                                          send_port))

        return s_receive, s_send


    def accept(self, s_recv, s_send):
        recv_conn, recv_addr = s_recv.accept()
        send_conn, send_addr = s_send.accept()
        
        print('Receiving connect by: ', recv_addr)
        print('Sending connect by: ', send_addr)

        return recv_conn, recv_addr, send_conn, send_addr


    def get_data(self, conn_receive, max_receive, q=None):
        data = conn_receive.recv(max_receive)

        # for single processing
        if q is None:
            return data

        # for multiprocessing 
        else:
            q.put(data)
            

    def bytes2str(self, data_bytes):
        data_decode = data_bytes.decode('utf-8')

        return data_decode


    def disconnect(self, s_receive, s_send):
        s_receive.shutdown(2)
        s_send.shutdown(2)
        s_receive.close()
        s_send.close()


class Socket3rdParty():
    def __init__(self, HOST, PORT):
        self.host = HOST
        self.port = PORT

        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 

        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.bind((HOST, PORT))
        self.s.listen(5)

#        self.s = self.connect()

#        self.conn = None
#        while self.conn is None:
#            self.conn = self.accept(self.s)


    def connect(self, inst):
#        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

#        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#        s.bind((host, port))
#        s.listen(5)

        while True:
            print('3rd server is waiting for connection ......')

            conn, addr = self.s.accept()
            inst.set(conn, addr)

            print('3rd Server start at: {}:{} for sending'.format(self.host, 
                                                                  self.port))
#            q.put((conn, addr))

        s.close()

    def accept(self, s, q=None):
        while True:
             conn, addr = s.accept()
 
             print('3rd-party-screen connect by: ', addr)
 
             if q is None:
                 return conn
 
             else:
                 q.put(conn)

    def disconnect(self):
        self.s.shutdown(2)
        self.s.close()

    def dummy_func(self):
        print('3rd linked.')


class SocketStatus():
    def __init__(self):
        self.conn = None
        self.addr = None

    def get(self):
        return self.conn, self.addr

    def set(self, conn, addr):
        self.conn = conn
        self.addr = addr



