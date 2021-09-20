import tornado.ioloop
import tornado.web
import tornado.httpserver
import tornado.escape
import src.article_classification as ac
import src.meta_classification as mc

import src.untrusted_sites as us
import json
import random, string
from tornado.httpclient import AsyncHTTPClient, HTTPRequest
import datetime
from tornado.web import MissingArgumentError
import time


httpList = []   # monitoring http traffic


class MainHandler(tornado.web.RequestHandler):

    def get(self, *args, **kwargs):
        self.render("index.html",  title="index")


class ContentDetectorHandler(tornado.web.RequestHandler):

    def get(self, *args, **kwargs):
        ac.init_params()
        self.render("content_detector.html", title="content_detector")


class ContentDetectorTabHandler(tornado.web.RequestHandler):

    def get(self, *args, **kwargs):
        self.render("content_detector_tab.html",  title="content_detector_tab")


class MetaDetectorHandler(tornado.web.RequestHandler):

    def get(self, *args, **kwargs):
        mc.init_params()
        self.render("meta_detector.html", title="meta_detector")


class MetaDetectorTabHandler(tornado.web.RequestHandler):

    def get(self, *args, **kwargs):
        self.render("meta_detector_tab.html",  title="meta_detector_tab")


class WordcloudHandler(tornado.web.RequestHandler):

    def get(self, *args, **kwargs):
        self.render("wordcloud.html",  title="wordcloud")


class UntrustedHandler(tornado.web.RequestHandler):

    def get(self, *args, **kwargs):
        us.load_untrusted()
        self.render("untrusted.html",  title="untrusted")


class MoreHandler(tornado.web.RequestHandler):

    def get(self, *args, **kwargs):
        self.render("more.html",  title="more")


class ContentQueryHandler(tornado.web.RequestHandler):

    def post(self, *args, **kwargs):

        try:
            if len(args) != 0:
                raise ValueError("Invalid URL")
            request = tornado.escape.json_decode(self.request.body)
            if "content" not in request:
                raise ValueError("missing version element")
            print("LOG :: ContentQueryHandler got content POST")

            q = request['content']
            # ac.load_model()
            # ac.test_model(q)
            ac.kr_load_model()
            ac.kr_predict(q)
        except KeyError as ex:
            self.send_error(404, message=ex)
        except ValueError as ex:
            self.send_error(400, message=ex)
        self.set_status(201, None)


class MetaQueryHandler(tornado.web.RequestHandler):

    def post(self, *args, **kwargs):

        try:
            if len(args) != 0:
                raise ValueError("Invalid URL")
            request = tornado.escape.json_decode(self.request.body)

            print("LOG :: MetaQueryHandler got content POST")

            q = request['title']
            mc.svm_load_model()
            mc.svm_predict(q, None)
            print(q)
        except KeyError as ex:
            self.send_error(404, message=ex)
        except ValueError as ex:
            self.send_error(400, message=ex)
        self.set_status(201, None)
