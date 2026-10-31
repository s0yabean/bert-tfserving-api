"""Gunicorn entry point: `gunicorn wsgi:app`."""
from bert_api import create_app

app = create_app()
