FROM python:3.6
EXPOSE 8501
WORKDIR /app
COPY reddit_classifier_app.py ./reddit_classifier_app.py
COPY requirements.txt ./requirements.txt
COPY ./model/reddit_classifier.h5 ./model/reddit_classifier.h5
COPY ./helpers/tokenize_text.py	./helpers/tokenize_text.py
COPY ./helpers/tokenizer_reddit.pkl ./helpers/tokenizer_reddit.pkl

RUN pip3 install -r requirements.txt
RUN python -m nltk.downloader punkt

CMD streamlit run reddit_classifier_app.py \
	--server.headless true \
    --browser.serverAddress="0.0.0.0" \
    --server.enableCORS true \
    --browser.gatherUsageStats false
