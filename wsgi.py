"""WSGI entry point for production (gunicorn on Render/Railway).

Gunicorn imports this module and looks for the ``app`` attribute. We build
it from the ``production`` config unless FLASK_ENV says otherwise, so
DEBUG is off and the real PostgreSQL DATABASE_URL is used.
"""

import os

from app import create_app

app = create_app(os.environ.get("FLASK_ENV", "production"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
