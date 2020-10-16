# Reddit Post Classifier
[Reddit](https://www.reddit.com/) is one of the largest online forums in the world with over **430 million** active users (according to [Statista](https://www.statista.com/forecasts/1174696/reddit-user-by-country)). People are heavily sharing (world) news, pictures, memes, facts, thoughts, controversies, and many more things on Reddit and chances are that there is already a specific Subreddit for a certain topic (e.g., checkout this awesome Subreddit all about the beauty of data: [/r/dataisbeautiful](https://www.reddit.com/r/dataisbeautiful/)).

However, where there is light, there must always be shadow. Misleading, manipulated or false connections can become a serious problem if not clearly recognized as such. Yet, the classification of these posts is not as simple for the user as it seems. In some cases, even the separation of satire and true news can indeed be a tricky task. What a time to live in!

That's why I've asked myself: Is there a way to use my enthusiasm for data and machine learning techniques to come up with a simple web application to help others classify Reddit posts? Turns out: Yes! There is even a whole web page (incl. publications) that focuses on the topic of Reddit post classification using multiple machine learning approaches and even provides training, test, and validation data for machine learning algorithms. Be sure to check out the awesome [Fakeddit](https://fakeddit.netlify.app) page.

But, since I'm always in for some fun with machine learning, natural language processing and cloud computation stuff, I wanted to come up with a model myself. Therefore, I split this project into smaller, digestible steps:

1. Clean up and prepare the Fakeddit data using **Amazon Web Services (AWS) SageMaker** (in theory, for larger datasets one should probably use **AWS Elastic Map Reduce (EMR)** and **Spark/Koalas**)
2. Specify and train/validate the model with the data also using **AWS SageMaker**
3. Create a local dashboard around the model for easy access using **Streamlit**
4. Wrap everything up in a **Docker** container
5. Push the container to the **AWS Elastic Container Registry (ECR)** and deploy the app using the **AWS Elastic Container Service (ECS)** and **Application Load Balancer (ALB)**
6. OPTIONAL: Use **AWS Route 53** and **AWS Cognito** for secure user authentication

Note that I used these resources to help me with the development, especially the deployment in AWS ECS: [[1]](https://github.com/nicolasmetallo/legendary-streamlit-demo), [[2]](https://github.com/aws-samples/aws-cdk-examples/tree/master/python/url-shortener), [[3]](https://dev.to/paulkarikari/build-train-and-deploy-tensorflow-deep-learning-models-on-amazon-sagemaker-a-complete-workflow-guide-495i), [[4]](https://aws-blog.de/2020/03/building-a-fargate-based-container-app-with-cognito-authentication.html), [[5]](https://blog.usejournal.com/using-nlp-to-detect-fake-news-289314fb9198).


So, let's get started and classify some Reddit posts. Here is an overview of what I have in mind:

![Reddit_Classifier_Overview](https://github.com/mhauck-FFM/Reddit_Post_Classifier/blob/main/Overview_Diagram.png)

## Data preparation

The first task to accomplish is the data cleansing and preparation. We need to identify the most important features of the training data and extract them into a separate dataset. Additionally, we have to include the *true* labels of the Reddit posts within the training data for our model to learn. The 6 labels are:

| Label                 | Index |
| --------------------- | ----- |
| True                  | 0     |
| Satire/Parody         | 1     |
| Misleading Content    | 2     |
| Manipulated Content   | 3     |
| False Connection      | 4     |
| Imposter Content      | 5     |

After that we have to tokenize the text into sequences of equal length which our model is able to understand. For this task, I'm using (of course) python 3.6 with pandas, keras, nltk, and pickle.

For the feature selection, I've decided that I want to keep it as simple and small as possible, so we will be training the model only on the ``clean_title`` of the Reddit post and use the ``6_way_label`` as our feature the model shall predict. This is a rough approximation and might lead to some strong biases as the model could, for example, learn that shorter titles are more likely to indicate that something with the post is wrong. However, I will start small and extend the features if need be later on.

The preprocessing of the data is also done in AWS SageMaker. I have started with a fancy, fully fledged python script that was called by a driver notebook to run the preprocessing on SageMaker and create a data pipeline. However, that was a total overkill. The processing steps are quite handleable so that a single python script that just runs on the notebook instance is fully sufficient. This script can be found in ``/preprocessing/reddit_data_preprocessing.py``. What it does is it loads the data from my S3 bucket, crams them into pandas dataframes, performs the feature selection and tokenization, and loads the final processed data as json files back to S3. As I said, nothing computationally intensive. For the tokenization, I use the tokenizer that comes with keras together with the ``texts_to_sequences`` and ``pad_sequences`` functions of nltk. Note that I had to use ``lambda`` functions and the pandas ``.apply()`` method to apply these functions to the complete dataframe without using slow loops explicitly. I'm using 25 words as sequence length, since Reddit titles are seldomly longer than that amount. The trained tokenizer is also saved to S3 using pickle for later use during prediction. For convenience, it can also be found in ``/helpers/tokenizer_reddit.pkl``. That's it. The data are now super clean.

## Model training

Now that we have clean data, we can work on the classification model. I decided to use a neural network with different Embedding, LSTM, Dropout, MaxPooling and Dense layers. As always with the design of neural networks, finding the exact, fine-tuned, final setup is kind of a trial and error process. I've tried various different layer combinations and complexities, but the final setup is quite simple. I also decided to incorporate a word2vec pretrained model to give the neural network some prior knowledge of word connections. The pretrained model is the 2013 Google News [Word2Vec](https://code.google.com/archive/p/word2vec/) with 300-dimensional vectors for 3 million words and phrases. The word2vec model is used in the embedding layer of the neural network and stored within the S3 bucket.

The script for the training can be found in ``/model_setup/train_reddit_classifier.py`` and the following graphic gives an overview of the model:

![model_overview](https://github.com/mhauck-FFM/Reddit_Post_Classifier/blob/main/model/reddit_classifier_plot.png)

For the activation functions of the LSTM layers I tested different approaches ('relu', 'tanh', 'sigmoid', ...), but a linear activation lead to the best performance during both training and testing. Note that I used the dataset called "test" as ``validation_data`` for fitting the model since in my thinking I test the performance of the model during the fitting and validate it afterwards. However, it doesn't really matter if using validation or test for that purpose. Just use one of it and NOT the training data!

The training process is done using a driver notebook that calls the script in ``/model_setup/train_reddit_classifier.py`` and creates a fully automated SageMaker training job that spins up the parsed instances, performs the training and saves the model to S3. Theoretically, we could even use SageMaker to deploy the model. But there are other tools that are suitable for that purpose (see below or above). I'm training the model for 20 epochs with a batch size of 16384 (a.k.a. 2^14). This might seem pretty large for a batch, but remember that we have almost 900000 rows in our training data that need to be processed in every epoch. Additionally, I wanted to keep it simple and cheap! After 20 epochs we achieve an accuracy of circa **94 %** during training and **92 %** during testing, which is not bad for such a relatively simple construct. The model is saved to S3 and amounts approximately 340 MB of space.

## Creating a Streamlit app

With the completion of the model training, we can now think about making predictions. Theoretically, I could just write a short python script where the title of the Reddit post is parsed and the probability for each category is returned to the console. **Spoiler alert**: this is what I did to test the general applicability of the new model and it works solidly. But of course, this is not the best way to deploy such a model, especially for users without any programming experience. So, we need to build an app for the model. And luckily, there is a python module that makes the development of web-based applications very smooth, even smoother than dash/plotly. I'm talking about the awesome [Streamlit](https://www.streamlit.io/).

With Streamlit we can create an application in under 100 cleanly formatted (**!**) rows of python code. But first, let's think about what features the app should have? Clearly, we need an *input box* where users can enter the title of the desired Reddit post they want to have classified. A neat extension would be the possibility to parse direct weblinks to the post and the app gets the title automatically. This is relatively simple considering the beautifulsoup library in python. And yes, I'm aware that there is a whole API for Reddit in python, but as always: keep it short and keep it simple.

Then, the app should *print* the *title* of the post (for convenience) together with the *probability* of the most likely category. And, of course, the *category name* as well. Finally, I would like to see the probability of all categories in a nifty *bar chart* below the printing part. So, we are going to add that, too. To increase the performance of the app, the model and tokenizer, which are essential for the predictions, are cached on the page. The code of the Streamlit app is stored in ``reddit_classifier_app.py``. And, et voilà, our app is done. We can test the app by running the underlying script in the console and access it via a browser under ``localhost:8501``. Works just fine.

Here is a picture of the app with an arbitrary Reddit post from [/r/theonion](https://www.reddit.com/r/TheOnion/comments/jazpb3/report_amtrak_loses_100_million_annually_to_route/). Of course, all The Onion posts are satire, which the model clearly detects:

![App_Example](https://github.com/mhauck-FFM/Reddit_Post_Classifier/blob/main/App_Example.png)

## Dockerizing the app

We are getting very close to the deployment of the app now. By now, Streamlit only runs in local mode and only if we open a specific terminal to run the script. For test purposes, this might be sufficient, but for real application it is not suitable. The best option is to create a web server, which runs the whole program. However, the app requires not only one script, but multiple files and python modules. And as a windows user, who often struggles with setting up linux programs on windows, I know just the perfect solution to this problem: Docker. Docker allows us to create a container, which contains (obviously) all files and programs our app requires to run. And all that in an easily deployable environment. Awesome, right? I know, I've been a fan of Docker ever since I first used it. To turn the Streamlit app and related files into a Docker container, we need two files: ``Dockerfile`` and ``requirements.txt``. The first one tells Docker what to do and the second one which python modules we want to include.

The ``Dockerfile`` looks as follows:
```
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
```

and the ``requirements.txt`` as follows:
```
tensorflow
keras
matplotlib
bs4
requests
lxml
numpy==1.18.5
pandas
streamlit
nltk
```

Then, simply change the working directory of your terminal to the folder containing all relevant files and run ``docker build -t NAME_OF_DOCKER_CONTAINER .``. Don't forget the dot after your specified container name! You can deploy your container by using ``docker run -it --rm --name NAME_OF_DOCKER_CONTAINER -p 8501:8501 NAME_OF_DOCKER_CONTAINER``. You can choose whatever name you fancy after ``--name``, but for consistency I almost always stick to the container name. Using ``localhost:8501`` you can now enter the app without having to run the python script in the terminal. And even better: you could now push your container to any Docker repository. That is exactly what we are going to do in the final step.

## Deployment on AWS ECS

For our final deployment of the app, we need to follow only a few more steps. First, we push our newly created Docker container to the AWS Elastic Container Registry. Note that you have to tag your local container properly following the AWS [guidelines](https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-push-ecr-image.html). For that, you need to authenticate your local Docker distribution with your AWS account. Again, follow the AWS [guidelines](https://docs.aws.amazon.com/AmazonECR/latest/userguide/Registries.html#registry_auth).

The next step is the one that was extremely new for me and took a whole lot of time to complete. But, getting it done and seeing your app for the first time publicly available is so satisfying. And I learned many useful new things along the way. So, don't be afraid. We are going to deploy the app using AWS Elastic Container Service on a Fargate Serverless Service. That is different from a deployment on a single EC2 instance, as we don't have to care about setting up the infrastructure by hand and especially the monitoring of the instance(s). Therefore, Fargate uses a so-called Application Load Balancer (ALB) to spin up EC2 instance(s) and handle all security and monitoring settings. If more computing power is necessary, the ALB will take care of it. For the ALB, we need to setup a ECS Service and Task. That sounds like a lot of work, but I'm using a neat toolkit to deploy all that stuff using python: the AWS Cloud Development Kit (CDK) - also known as Infrastructure as Code (IaC).

CDK creates a stack in which you specify the AWS VPC, Fargate Service, and Task and the code sets everything up - ready to go for you. Just look at this awesome [tutorial](https://aws-blog.de/2020/03/building-a-fargate-based-container-app-with-cognito-authentication.html). I'm not going to show some code here, as it really helps to get your hands on it to understand what happens with your code.

That's it! Our app now runs in the cloud and can be accessed using the public IP address of the Load Balancer. Theoretically, we could assign a specific fancy domain (e.g., www.reddifier.com) for the app and route it to the ALB IP using Route 53. And even more fancy, we can use AWS Cognito to implement user authentication. But, that is of course charged separately by AWS.
