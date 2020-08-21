''' Step 1 in ML:
Dividing the dataset(both features & labels) into training & test



'''
#Classification using logistic regression

import nltk
nltk.download('twitter_samples')
nltk.download('stopwords')

'''
2 important functions in entire code:
Both functions are present in utils library
1. process_tweet() -> removes handles,stopwords .. tokenizes and gives stem of 
tokenized words

2. build_freqs() -> gives frequency of a specific word in positive and negative
tweets. then builds the freqs dict where (word,label) is key & corresponding 
frequency is value

'''

####Packages####

import numpy as np #for math functions like log,dot,exp
import pandas as pd
from nltk.corpus import twitter_samples
from utils import process_tweet, build_freqs

####Packages END####

### Splitting the data into training(80) & test(20)###

pos_tweets=twitter_samples('positive_tweet.json')
neg_tweets=twitter_samples('negative_tweet.json')
##info-> total tweets=5000
##80% is 4000 tweets(+ve) in training

train_x=pos_tweets[:4000] + neg_tweets[:4000]
test_x=pos_tweets[4000:] + neg_tweets[4000:]

## Here, tweets are 'features' and 'labels' are 1(+ve) and 0(-ve)

### Now we need to define labels for above features
##combine both positive & negative labels
train_y=np.append(np.ones((4000,1)),np.zeros((4000,1)),axis=0)
test_y=np.append(np.ones((1000,1)),np.zeros((1000,1)),axis=0)

'''
Things that we don't haveto do:
1. As process_tweet() gives us tokenized stem we don't have to do
any of the preprocessing steps.
2. Now we are interested in frequency of words for extracting features 
and not the tokenized stems. As build_freqs() directly gives us the frequency 
count we  don't even have to call process_tweet() as build_freqs() does
it implicitly

Parameters of build_freq:
1. tweet(i. feature)
2. Label associated with tweet 

Output:
returns a dict where (word,label) is 'key' & frequency is 'value' 
'''

###Create frequency dictionary
freqs=build_freqs(train_x,train_y)

#### Now, logistic regression starts
'''Use of sigmoid function

	ℎ(𝑧)=1/(1+exp−𝑧)

if h(z)>0.5 then label prediction=1
else label prediction=0

But what is z?   z = np.dot(x,theta)

So lets find theta in coming step and then x
'''

### sigmoid function

def sigmoid(z):
	h=1/(1+np.exp(-z))
	return h

###Cost function & gradient
'''
Cost function to check the error in predicted value
Less the cost, more is the accuracy

𝐽(𝜃)=−1/𝑚*(∑𝑦(𝑖)log(ℎ(𝑧(𝜃)(𝑖)))+(1−𝑦(𝑖))log(1−ℎ(𝑧(𝜃)(𝑖))))

We need 2 things:
1. Feature vector(i.e. x) of order 3*1 [bias, summn(positive freq), summn(negative freq)]
2. weight vector(theta)

Now for finding theta we have to do iterations for getting right values

Following are the steps to find theta:
Cost function of gradient

	∇𝜃𝑗𝐽(𝜃)=1/𝑚*(∑(ℎ(𝑖)−𝑦(𝑖))*𝑥𝑗)

𝐽=−1/𝑚×(𝐲𝑇⋅𝑙𝑜𝑔(𝐡)+(1−𝐲)𝑇⋅𝑙𝑜𝑔(1−𝐡))

Updated value of theta after every iteration
	𝜃𝑗=𝜃𝑗−𝛼×∇𝜃𝑗𝐽(𝜃)
The learning rate  𝛼  is a value that
 we choose to control how big a single update will be

x(j) is the feature associated with theta(j)
'''

###Gradient Descent Function
'''
1.Calculate cost function
-> for this we need h(z)-> so we need z
i)calculate z
ii)calculate h(z) 

2.Update weights
'''
def gradientDescent(x,y,theta,alpha,um_iters):
	#here theta that we pass in function is initial asumption
	z=np.dot(x,theta)
	h=sigmoid(z)

	#cost function
	#𝐽=−1/𝑚×(𝐲𝑇⋅𝑙𝑜𝑔(𝐡)+(1−𝐲)𝑇⋅𝑙𝑜𝑔(1−𝐡))
	
	yT=np.transpose(y)
        a=[(1-y[i]) for i in range(len(y))]
        b=[(1-h[i]) for i in range(len(h))]
        aT=np.transpose(a)
        j1=np.dot(yT,np.log(h))
        j2=np.dot(aT,np.log(b))
        J = -(1/m)*(j1+j2)

	#Update the weights theta

	xT=np.transpose(x)
        diff=[(h[i]-y[i]) for i in range(len(h))]
        theta = theta-((alpha/m)*(np.dot(xT,diff)))

	J=float(J)
	return J,theta # J to see how accurate is value if theta

## now we have found theta after lot of training

## now we have to extract features i.e find x 
		
def extract_features(tweet,freqs):
	
	word_l=process_tweet(tweet)
	x=np.zeros((1,3))
	x[0,0]=1

	for word in word_l:
		key0=(word,0)
		key1=(word,1)
		#negative label frequency
		if(key0 in freqs.keys()):
			x[0,2]+=freqs[key0]
		#positive label frequency
		if(key1 in freqs.keys()):
			x[0,1]+=freqs[key1]
	assert(x.shape==(1,3))  #checking whether elements of x is of order 1*3
	return x
###We have found x. Now, lets predict tweet in next steps

'''Hey wait... Although we have defined gradient descent we haven't found
theta for our training set yet
 Let's find theta for our training set
Calling all funtions defined until now
'''
####part 3 : Training model

X=np.zeros((1,3))
for i in range(len(train_x)):
	X[i:]=extract_features(train_x[i],freqs)
Y=train_y

theta=np.zeros((3,1))
alpha=1e-9
iterations=1500

J,theta=gradientDescent(X,Y,theta,alpha,iterations)

### Finally predict_tweet

def predict_tweet(tweet,freqs,theta):
	## we need x and theta(function param) 
	## then call sigmoid function
	x=extract_features(tweet,freqs)
	z=np.dot(x,theta)
	prediction=sigmoid(z)

	return prediction


### finally done with prediction


'''Last step:
Checking performance of our model:
'''
def test_logistic_regression(test_x, test_y, freqs, theta):
    """
    Input: 
        test_x: a list of tweets
        test_y: (m, 1) vector with the corresponding labels for the list of tweets
        freqs: a dictionary with the frequency of each pair (or tuple)
        theta: weight vector of dimension (3, 1)
    Output: 
        accuracy: (# of tweets classified correctly) / (total # of tweets)
    """
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    # the list for storing predictions
    y_hat = []
    
    for tweet in test_x:
        # get the label prediction for the tweet
        y_pred = predict_tweet(tweet, freqs, theta)
        
        if y_pred > 0.5:
            # append 1.0 to the list
            y_hat.append(1)
        else:
            # append 0 to the list
            y_hat.append(0)

    # With the above implementation, y_hat is a list, but test_y is (m,1) array
    # convert both to one-dimensional arrays in order to compare them using the '==' operator
    test_y=list(test_y)
    
    acc = [(1 if y_hat[i]==test_y[i] else 0) for i in range(len(y_hat))]
    accuracy=0
    for i in acc:
        accuracy=accuracy+i
    accuracy=accuracy/len(test_x)
    ### END CODE HERE ###
    
    return accuracy

### Byeeeee ;)


