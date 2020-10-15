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


Here is an overview of what I have in mind:
![Reddit_Classifier_Overview](https://github.com/mhauck-FFM/Reddit_Post_Classifier/blob/main/Overview_Diagram.png)

So, let's get started and classify some Reddit posts.

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

The preprocessing of the data is also done in AWS SageMaker. I have started with a fancy, fully fledged python script that was called by a driver notebook to run the preprocessing on SageMaker and create a data pipeline. However, that was a total overkill. The processing steps are quite handleable so that a single python script that just runs on the notebook instance is fully sufficient. This script can be found in ``/preprocessing/reddit_data_preprocessing.py``. What it does is it loads the data from my S3 bucket, crams them into pandas dataframes, performs the feature selection and tokenization, and loads the final processed data as json files back to S3. As I said, nothing computationally intensive. For the tokenization, I use the tokenizer that comes with keras together with the ``texts_to_sequences`` and ``pad_sequences`` functions of nltk. Note that I had to use ``lambda`` functions and the pandas ``.apply()`` method to apply these functions to the complete dataframe without using slow loops explicitly. The trained tokenizer is also saved to S3 using pickle for later use during prediction. That's it. The data are now super clean.

## Model training

## Creating a Streamlit app

## Dockerizing the app

## Deployment on AWS ECS
