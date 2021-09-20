import os.path

# Paths
DIRNAME = os.path.dirname(__file__)
ROOT_PATH = os.path.normpath(os.path.join(DIRNAME, '..'))
STATIC_PATH = os.path.join(DIRNAME, 'static')
TEMPLATE_PATH = os.path.join(DIRNAME, 'templates')

mainDataSetPath = '/home/nik/PycharmProjects/fake_news/Datasets'
modelPath = '/home/nik/PycharmProjects/fake_news/models'

# ConfigDB
CONFIGDB_PATH = "%s/db_deploy/db/fake_news.db" % (ROOT_PATH,)
CONFIGDB_ENGINE = "sqlite:///%s" % (CONFIGDB_PATH,)


# import base64
# import uuid
# COOKIE_SECRET = base64.b64encode(uuid.uuid4().bytes + uuid.uuid4().bytes)
COOKIE_SECRET = b'xyRTvZpRSUyk8/9/McQAvsQPB4Rqv0w9mBtIpH9lf1o='
DEBUG = True

MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join('/fake_news', 'media')
