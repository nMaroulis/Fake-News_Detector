from sqlalchemy.exc import IntegrityError
from db_deploy.db_manager import Session, News_articles
from sqlalchemy import func
import os

titleHashMap = {}


def generate_article_table():

    if not Session().query(News_articles).all():

        print("LOG: News Articles Table")
        session = Session()
        session.add(News_articles(article_id="Null entry", article_title= "Null entry", article_body="Null Entry"))
        session.commit()


class Core(object):

    def __init__(self):

        self.ptitles = {}
        generate_article_table()

    def add_articles(self):
        session = Session()

        rows = session.query(func.count(News_articles.article_id)).scalar()
        if rows <= 1:
            try:
                for k, v1, v2 in titleHashMap.items():
                    request = News_articles(article_id=str(k), article_title=str(v1), article_body=str(v2))
                    session.add(request)
                    print(k + " ----- " + str(v1))
                session.commit()

            except IntegrityError:
                session.rollback()
                raise ValueError("Value Error on session Commit")
        else:
            print("LOG: add_articles had no action because table already has " + str(rows) + " rows.")


class NewsArticle(object):

    def __init__(self, article_id, article_title, article_body):
        self._article_id = article_id
        self._article_title = article_title
        self._article_body = article_body

    def to_dict(self):
        """Return JSON-serializable representation of the object."""
        return {'article_id': self._article_id, 'article_title': self._article_title, 'article_body': self._article_body}
