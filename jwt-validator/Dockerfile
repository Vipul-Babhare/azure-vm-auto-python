FROM python:3.9-slim

WORKDIR /app
COPY app.py .
RUN pip install flask pyjwt requests

ENV SECRET_KEY=my_super_secret_key
EXPOSE 8000

CMD ["python", "app.py"]
