# Reddit Post Classifier
[Reddit](https://www.reddit.com/) is one of the largest online forums in the world with over **430 million** active users (according to [Statista](https://www.statista.com/forecasts/1174696/reddit-user-by-country)). People are heavily sharing (world) news, pictures, memes, facts, thoughts, controversies, and many more things on Reddit and chances are that there is already a specific Subreddit for a certain topic (e.g., checkout this awesome Subreddit all about the beauty of data: [/r/dataisbeautiful](https://www.reddit.com/r/dataisbeautiful/)).

However, where there is light, there must always be shadow. Misleading, manipulated or false connections can become a serious problem if not clearly recognized as such. Yet, the classification of these posts is not as simple for the user as it seems. In some cases, even the separation of satire and true news can indeed be a tricky task.

That's why I've asked myself: Is there a way to use my enthusiasm for data and machine learning techniques to come up with a simple web application to help others classify Reddit posts? Turns out: Yes! There is even a whole web page (incl. publications) that focuses on the topic of Reddit post classification using multiple machine learning approaches and even provides training, test, and validation data for machine learning algorithms. Be sure to check out the awesome [Fakeddit](https://fakeddit.netlify.app) page. But, since I'm always in for some fun with machine learning, natural language processing and cloud computation stuff, I wanted to come up with a model myself. Therefore, I split this project into smaller, digestible steps:

1. Clean up and prepare the Fakeddit data using **Amazon Web Services** (AWS) **SageMaker** (in theory, for larger datasets one should probably use **AWS Elastic Map Reduce (EMR)** and **Spark/Koalas**)
2. Specify and train/validate the model with the data also using **AWS SageMaker**
3. Create a local dashboard around the model for easy access using **Streamlit**
4. Wrap everything up in a **Docker** container
5. Push the container to the **AWS Elastic Container Registry (ECR)** and deploy the app using the **AWS Elastic Container Service (ECS)** and **Application Load Balancer (ALB)**
6. OPTIONAL: Use **AWS Route 53** and **AWS Cognito** for secure user authentication

Note that I used these resources to help me with the development, especially the deployment in AWS ECS: [[1]](https://github.com/nicolasmetallo/legendary-streamlit-demo), [[2]](https://github.com/aws-samples/aws-cdk-examples/tree/master/python/url-shortener), [[3]](https://dev.to/paulkarikari/build-train-and-deploy-tensorflow-deep-learning-models-on-amazon-sagemaker-a-complete-workflow-guide-495i), [[4]](https://blog.usejournal.com/using-nlp-to-detect-fake-news-289314fb9198).

So, let's classify some Reddit posts.

## Data preparation and model training
