import os
import sys
from operator import add
# from pyspark import *
import threading
import src.core.entities
import json
import os
from src.rest_services.handlers import MainHandler, ContentDetectorHandler, ContentDetectorTabHandler, MetaDetectorHandler, MetaDetectorTabHandler, ContentQueryHandler, MetaQueryHandler, WordcloudHandler, UntrustedHandler, MoreHandler
import tornado.escape
import tornado.httpserver
import tornado.ioloop
import tornado.web
from tornado.httpclient import AsyncHTTPClient, HTTPRequest
from settings import TEMPLATE_PATH, STATIC_PATH, DEBUG, COOKIE_SECRET
from core.entities import Core
import datetime
from tornado import gen
from tornado.concurrent import run_on_executor
from src.article_classification import *

# os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3.6"
SERVER_IP = 'http://127.0.0.1'
SERVER_PORT = 1234


def spark_mapreduce():
    print('asdsad')


def main():
    return


def make_app():
    core = Core()
    parms = {
        "template_path": TEMPLATE_PATH,
        "static_path": STATIC_PATH,
        "debug": DEBUG,
        "cookie_secret": COOKIE_SECRET,
    }
    return tornado.web.Application([(r"/", MainHandler),
                                    (r"/content_detector", ContentDetectorHandler),
                                    (r"/content_detector_tab", ContentDetectorTabHandler),
                                    (r"/meta_detector", MetaDetectorHandler),
                                    (r"/meta_detector_tab", MetaDetectorTabHandler),
                                    (r"/content_query_handler", ContentQueryHandler),
                                    (r"/meta_query_handler", MetaQueryHandler),
                                    (r"/wordcloud", WordcloudHandler),
                                    (r"/untrusted", UntrustedHandler),
                                    (r"/more", MoreHandler),
                                    ], **parms)


def loop():
    print(" loop (" + datetime.datetime.now().strftime("%H:%M:%S") + ")")
    pass


if __name__ == "__main__":

    # executor = ThreadPoolExecutor(max_workers=4)
    print("LOG :: Starting Server")
    app = make_app()
    http_server = tornado.httpserver.HTTPServer(app)
    app.listen(SERVER_PORT)
    ioloop = tornado.ioloop.IOLoop.instance()     # task = tornado.ioloop.PeriodicCallback(loop, 30000)
    main_task = threading.Thread(target=main)
    main_task.start()
    print("LOG :: Server Running at " + SERVER_IP + ':' + str(SERVER_PORT))
    # task.start()
    ioloop.start()
