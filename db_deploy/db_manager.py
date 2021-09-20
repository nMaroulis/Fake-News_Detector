from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Integer, ForeignKey, FLOAT, LargeBinary, Boolean
from sqlalchemy.types import TypeDecorator, Unicode
from sqlalchemy import create_engine, event
from sqlalchemy.orm import scoped_session, sessionmaker
import os.path
# from settings import CONFIGDB_ENGINE

DIRNAME = os.path.dirname(__file__)
ROOT_PATH = os.path.normpath(os.path.join(DIRNAME, '..'))
STATIC_PATH = os.path.join(DIRNAME, 'static')
TEMPLATE_PATH = os.path.join(DIRNAME, 'templates')
# ConfigDB
CONFIGDB_PATH = "%s/db_deploy/db/fake_news.db" % (ROOT_PATH,)
CONFIGDB_ENGINE = "sqlite:///%s" % (CONFIGDB_PATH,)

ENGINE = create_engine(CONFIGDB_ENGINE, pool_recycle=6000)


def on_connect(conn, record):
    conn.execute('pragma foreign_keys=ON')


event.listen(ENGINE, 'connect', on_connect)
SESSION_FACTORY = sessionmaker(autoflush=True,
                               bind=ENGINE,
                               expire_on_commit=False)


Session = scoped_session(SESSION_FACTORY)

Base = declarative_base()


class News_articles(Base):

    __tablename__ = 'news articles'

    article_id = Column(String, primary_key=True)
    article_title = Column(String)
    article_body = Column(String)


Base.metadata.create_all(ENGINE)
