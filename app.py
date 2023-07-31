from distutils.log import debug
from fileinput import filename
import pandas as pd
from flask import *
import os
from werkzeug.utils import secure_filename
import numpy as np
import re
import csv
import nltk
import string
from nltk import FreqDist
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm
from wordcloud import WordCloud
from collections import Counter

UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
 
# Define allowed files
ALLOWED_EXTENSIONS = {'csv'}
 
app = Flask(__name__)
 
# Configure upload file path flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
app.secret_key = 'Bismillah Skripsi'


@app.route("/")
def index():
    return render_template("index.html")
'''
@app.route("/data_berlabel")
def labelNB():
    data = pd.read_csv("data berlabel.csv",encoding='unicode_escape')
    data = data[['sentimen', 'text']]
    #data_html = data.to_html()
    return render_template("show_data_berlabel.html", data=data)
'''

@app.route('/upload/data_unlabel', methods=['GET', 'POST'])
def uploadFile():
    if request.method == 'POST':
      # upload file flask
        f_test = request.files.get('file')
 
        # Extracting uploaded file name
        data_unlabel_filename = secure_filename(f_test.filename)
 
        f_test.save(os.path.join(app.config['UPLOAD_FOLDER'],data_unlabel_filename))
 
        session['uploaded_data_unlabel_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_unlabel_filename)
 
        return render_template('index2.html')
    return render_template("index3.html")


@app.route('/show_data_unlabel')
def showDataUnlabel():
    # Uploaded File Path
    data_unlabel_file_path = session.get('uploaded_data_unlabel_path', None)
    # read csv
    uploaded_df = pd.read_csv(data_unlabel_file_path, encoding='unicode_escape')
    # Converting to html Table
    #uploaded_df_html = uploaded_df.to_html()
    return render_template('show_data_unlabel.html', data=uploaded_df.to_dict(orient='records'))


# fungsi NB
def LabelNB(df_label, df_unlabel):
    #data label
    df = df_label.copy(deep=True)
    df.drop_duplicates(subset=['text'], inplace=True)
    df.dropna(subset=['text'], inplace=True)
    # Convert the data types to string
    df['text'] = df['text'].astype(str)
    label_encoder = {'negatif': 'negative', 'positif': 'positive','netral': 'neutral'}
    df['sentimen'] = df['sentimen'].map(label_encoder)
    df = df[df['sentimen'] != 'sentimen']    # drop the rows containing 'confused' labels
    df = df[['sentimen', 'text']]

	# START Prosessing
    # # 1. Clean the text
    def tweet_cleaner(tweet):
        stopwords = ['rt','rts', 'retweet', 'quot', 'sxsw']
        punctuation = set(string.punctuation) # punctuation 
        punctuation.remove('#') # remove # so hashtags remain in x
        x = tweet
        x = re.sub(r'https?:\/\/\S+', '', x) # remove URL references
        x = re.sub(r'{link}', '', x)  # remove placeholders
        x = re.sub(r'@[\w]*', '', x) # remove @mention users
        x = re.sub('[^A-Za-z0-9]+', ' ', x) # remove @mention users
        x = re.sub(r'\b[0-9]+\b', '', x) # remove stand-alone numbers
        x = re.sub(r'&[a-z]+;', '', x) # remove HTML reference characters
        x = ''.join(ch for ch in x if ch not in punctuation) # remove punctuation
        x = x.replace("[^a-zA-z#]", " ")  #remove special characters
        x = [word.lower() for word in x.split() if word.lower() not in stopwords]
        x = [w for w in x if len(w)>2]
        return ' '.join(x)
    
    df['clean_tweets1'] = df['text'].apply(tweet_cleaner)
    FreqDist(df['clean_tweets1'].unique().sum().split())
    
    # 2. stopword
    from nltk.corpus import stopwords
    indo_stopwords = stopwords.words('indonesian')
    def remove_stopwords(tweet):
        stopwords_removed = [word for word in tweet.split() if word not in indo_stopwords]
        return ' '.join(stopwords_removed)
    
    df['clean_tweets2'] = df['clean_tweets1'].apply(remove_stopwords)
    FreqDist(df['clean_tweets2'].unique().sum().split())
    
    # 3. Lemmatization
    from nlp_id.lemmatizer import Lemmatizer
    lemmatizer = Lemmatizer()
    def lemma(tweet):
        tweet=lemmatizer.lemmatize(tweet)
        return tweet
    
    df['clean_tweets3'] = df['clean_tweets2'].apply(lemma)
    FreqDist(df['clean_tweets3'].unique().sum().split())

    # Modeling
    X = df[['clean_tweets3']]
    y = df['sentimen']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y,random_state=20)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train,random_state=20)
    X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape
    
    from sklearn.pipeline import Pipeline
    estimators=[('vectorizer', TfidfVectorizer())]
    preprocessing_pipeline=Pipeline(estimators)	
    # Vectorizing
    X_train_vec = preprocessing_pipeline.fit_transform(X_train['clean_tweets3'])
    X_test_vec = preprocessing_pipeline.transform(X_test['clean_tweets3'])
    X_val_vec = preprocessing_pipeline.transform(X_val['clean_tweets3'])

    clf_NB = MultinomialNB()
    clf_NB.fit(X_train_vec, y_train)
    
    #data unlabel
    # Create a working dataframe with easier column name
    df_predict = df_unlabel.copy(deep=True)
    df_predict.drop_duplicates(subset=['text'], inplace=True)
    df_predict.dropna(subset=['text'], inplace=True)
    # Convert the data types to string
    df_predict['text'] = df_predict['text'].astype(str)
    df_predict = df_predict[['text']]

    # Apply the tweet cleaner to whole dataframe
    df_predict['clean_tweets1'] = df_predict['text'].apply(tweet_cleaner)
    # Word count of all the vocabulary
    FreqDist(df_predict['clean_tweets1'].unique().sum().split())

    df_predict['clean_tweets2'] = df_predict['clean_tweets1'].apply(remove_stopwords)
    FreqDist(df_predict['clean_tweets2'].unique().sum().split())

    df_predict['clean_tweets3'] = df_predict['clean_tweets2'].apply(lemma)
    FreqDist(df_predict['clean_tweets3'].unique().sum().split())

    X_uji = df_predict[['clean_tweets3']]
    X_uji_vec = preprocessing_pipeline.transform(X_uji['clean_tweets3'])
    
    #prediksi data uji
    prediksi=clf_NB.predict(X_uji_vec)
    X_uji.insert(1, column='label_bayes', value=prediksi)
    df_predict.insert(0, column='sentimen', value=prediksi)

    df['year'] = '2021'
    df_predict['year'] = '2022'

    df = df.append(df_predict)
    df.rename(columns = {'sentimen':'sentimen', 'text':'text','clean_tweets1':'hasil_cleaning','clean_tweets2':'hasil_stop_word','clean_tweets3':'hasil_lemmatize'}, inplace=True)
    
    return df


@app.route('/data/Labeling_NB')
def NB():
    # Uploaded File Path
    data_unlabel = session.get('uploaded_data_unlabel_path', None)
    
    # read csv
    df_label = pd.read_csv("data berlabel.csv", encoding='unicode_escape')
    df_unlabel = pd.read_csv(data_unlabel, encoding='unicode_escape')
    
    data_unlabel_predict = LabelNB(df_label, df_unlabel)
    data_to_csv = data_unlabel_predict
    data_to_csv.to_csv("data hasil labeling.csv",index=False)

    #uploaded_df_predict_html = data_unlabel_predict.to_html()

    return render_template('show_data_labeling.html', data=data_unlabel_predict.to_dict(orient='records'))
    #return df
    

@app.route('/model_LSTM')
def ModelLSTM():
      
    # read csv
    data = pd.read_csv('data hasil labeling.csv', encoding='unicode_escape')

    df = data
    import warnings
    warnings.filterwarnings('ignore')

    X = df['hasil_lemmatize']
    y_ohe = pd.get_dummies(df['sentimen'])

    X_train, X_test, y_train, y_test = train_test_split(X, y_ohe, test_size=0.10, stratify=y_ohe, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train, random_state=42)

    # Tokenization
    def create_tokens(X_train, X_val, X_test):
        '''
        A simple function to create word tokens with padded sequences
        '''
        tokenizer = Tokenizer(oov_token=True)
        tokenizer.fit_on_texts(X_train)

        X_train_token = tokenizer.texts_to_sequences(X_train)
        X_test_token = tokenizer.texts_to_sequences(X_test)
        X_val_token = tokenizer.texts_to_sequences(X_val)

        vocab_size = len(tokenizer.word_index) + 1

        maxlen = len(max(X_train_token, key=lambda x: len(x)))
        maxlen_orig= len(max(X_train, key=lambda x: len(x)))

        X_train_seq = pad_sequences(X_train_token, padding='post', maxlen=maxlen)
        X_test_seq = pad_sequences(X_test_token, padding='post', maxlen=maxlen)
        X_val_seq = pad_sequences(X_val_token, padding='post', maxlen=maxlen)

        print(f"Token count: {tokenizer.document_count}, Vocab size: {vocab_size}, Max lenth: {maxlen}, Original length: {maxlen_orig}")
        
        return X_train_seq, X_test_seq, X_val_seq, maxlen, vocab_size, tokenizer
    
    X_train_seq, X_test_seq, X_val_seq, maxlen, vocab_size, tokenizer = create_tokens(X_train, X_val, X_test)

    # Modeling
    def graph_model(history, metrics):
        plt.plot(history.history[metrics])
        plt.plot(history.history['val_'+ metrics])
        plt.xlabel('Epochs')
        plt.ylabel(metrics)
        plt.legend(['training', 'test'], loc='upper right')
        plt.show()
    
    # Use GLOVE pretrained model
    import os
    embeddings_index = {}
    f = open(os.path.join('glove.6B.100d.txt'), encoding='utf8')
    for line in tqdm(f):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    # Create a weight matrix for work vocabulary from our training set
    embedding_matrix = np.zeros((vocab_size, 100))  # 100 for 100-dimensional version
    for word, i in tqdm(tokenizer.word_index.items()):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    def predict_df(y_true, y_pred):
        '''
        A simple function to put predicted results into a dataframe
        '''
        true_df = pd.DataFrame(y_true.idxmax(axis=1), columns=['true_emotion']).reset_index(drop=True)
        pred_df = pd.DataFrame(y_pred.argmax(axis=1), columns=['predicted'])
        pred_df['predicted'] = pred_df['predicted'].apply(lambda x: 'negative' if x==0
                                                     else 'neutral' if x==1
                                                     else 'positive' )
        merge_df = pd.merge(true_df, pred_df, left_index=True, right_index=True)
        print(classification_report(merge_df['true_emotion'], merge_df['predicted']))
        
        return merge_df
    
    y_ohe_numpy = y_ohe.to_numpy()

    from sklearn.utils.class_weight import compute_class_weight
    y_integers = np.argmax(y_ohe_numpy, axis=1)
    class_weights = compute_class_weight(class_weight = "balanced",classes = np.unique(y_integers), y = y_integers)
    d_class_weights = dict(enumerate(class_weights))
    d_class_weights

    # Define callbacks and save final model
    def predict_w(model, epochs, batch_size, weights):
        early_stop = [EarlyStopping(monitor='val_loss', patience=5),
             ModelCheckpoint(filepath='best_model_m1.h5', monitor='val_loss',
                            save_best_only=True)]
        
        history = model.fit(X_train_seq, y_train,
                     batch_size=batch_size, epochs=epochs, verbose=1,
                     validation_data=(X_val_seq, y_val),
                     callbacks=early_stop,
                     class_weight=weights)
        
        graph_model(history, 'loss')
        graph_model(history, 'accuracy')

        train_prediction = model.predict(X_train_seq, batch_size=batch_size)
        val_prediction = model.predict(X_val_seq, batch_size=batch_size)
        test_prediction = model.predict(X_test_seq, batch_size=batch_size)

        return history, train_prediction, val_prediction, test_prediction
    
    # Final Modeling
    def create_model(model_type, vocab_size, maxlen, embed_dim):
        '''
        Create and return a compiled model
        '''

        if model_type == 'lstm':
            model = Sequential()
            model.add(layers.Embedding(input_dim = vocab_size, output_dim = 100,
                             weights=[embedding_matrix],
                             input_length=maxlen, trainable=False))
            model.add(layers.Dropout(0.4))
            model.add(layers.Bidirectional(layers.LSTM(embed_dim, return_sequences=True)))
            model.add(layers.GlobalMaxPool1D())
            model.add(layers.Dropout(0.2))
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dense(3, activation='softmax'))

            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model
    
    final_model = create_model('lstm', vocab_size, maxlen, embed_dim=128)
    final_model.summary()

    # Model fit
    checkpoint_path = os.path.join('lstm_model.h5')
    
    # Create a Model Check point
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=False, 
        monitor='loss', 
        verbose=1, 
        mode='auto', 
        save_best_only=True)
    
    history_final = final_model.fit(X_train_seq, y_train,
                          batch_size=32,
                          epochs=20,
                          verbose=1,
                          validation_data=(X_val_seq, y_val),
                          callbacks= [checkpoint])
    
    train_prediction = final_model.predict(X_train_seq, batch_size=32)
    val_prediction = final_model.predict(X_val_seq, batch_size=32)
    test_prediction = final_model.predict(X_test_seq, batch_size=32)
    
    #graph_model(history_final, 'loss')
    #graph_model(history_final, 'accuracy')

    train_predict_df = predict_df(y_train, train_prediction)
    val_predict_df = predict_df(y_val, val_prediction)

    # Prediction
    def evaluate_nn(model, X_test, y_test):
        """Print model accuracy on test set."""

        loss, acc = model.evaluate(X_test, y_test)
        print(f'Model Accuracy:\n\t{round(acc, 3)}')
    
    ## Evaluate model
    evaluate_nn(final_model, X_test_seq, y_test)

    test_prediction = final_model.predict(X_test_seq, batch_size=32)
    test_predict_df = predict_df(y_test, test_prediction)
    # Examples of Test Prediction
    X_test_orig = X_test.copy(deep=True)
    X_test_orig.reset_index(drop=True, inplace=True)
    X_test_orig = pd.DataFrame(X_test_orig.values, columns=['tweet'])
    X_test_pred_merge = pd.merge(X_test_orig, test_predict_df, left_index=True, right_index=True)

    train_prediction = final_model.predict(X_train_seq, batch_size=32)
    train_predict_df = predict_df(y_train, train_prediction)
    X_train_orig = X_train.copy(deep=True)
    X_train_orig.reset_index(drop=True, inplace=True)
    X_train_orig = pd.DataFrame(X_train_orig.values, columns=['tweet'])
    X_train_pred_merge = pd.merge(X_train_orig, train_predict_df, left_index=True, right_index=True)

    val_prediction = final_model.predict(X_val_seq, batch_size=32)
    val_predict_df = predict_df(y_val, val_prediction)
    X_val_orig = X_val.copy(deep=True)
    X_val_orig.reset_index(drop=True, inplace=True)
    X_val_orig = pd.DataFrame(X_val_orig.values, columns=['tweet'])
    X_val_pred_merge = pd.merge(X_val_orig, val_predict_df, left_index=True, right_index=True)

    X_train_pred_merge = X_train_pred_merge.append(X_val_pred_merge)
    X_train_pred_merge = X_train_pred_merge.append(X_test_pred_merge)

    X_train_pred_merge.to_csv("data hasil lstm.csv",index=False)
    #X_train_pred_merge_html = X_train_pred_merge.to_html()

    def create_wordcloud(data, col, label):
        wordcloud = WordCloud(background_color='white').generate(str(col))
        wordcloud.to_file("static/"+label+".png")
    
    create_wordcloud(X_train_pred_merge.loc[X_train_pred_merge['predicted']=='negative'], X_train_pred_merge['tweet'],"negative")
    create_wordcloud(X_train_pred_merge.loc[X_train_pred_merge['predicted']=='positive'], X_train_pred_merge['tweet'],"positive")
    create_wordcloud(X_train_pred_merge.loc[X_train_pred_merge['predicted']=='neutral'], X_train_pred_merge['tweet'],"neutral")

    return render_template('show_data_lstm.html', data=X_train_pred_merge.to_dict(orient='records'))

@app.route("/data/analisis", methods=['GET'])
def getDataLSTM():
    data = pd.read_csv('data hasil lstm.csv')
    sum_positive = sum(data.predicted  == 'positive')
    sum_neutral = sum(data.predicted  == 'neutral')
    sum_negative = sum(data.predicted  == 'negative')

    if(request.method == 'GET'):
        data = {
            'positive' : sum_positive,
            'neutral' : sum_neutral,
            'negative' : sum_negative
        }
    
    return jsonify(data)

@app.route("/analisis")
def analisis():
    data = pd.read_csv("data hasil lstm.csv",encoding='unicode_escape')
    
    pos = data.loc[data['predicted']=='positive']
    Counter_pos  = Counter(pos['tweet'])
    most_words_pos = Counter_pos.most_common(5)
    dt_pos = pd.DataFrame(most_words_pos)
    dt_pos.columns = ['postive_tweet','count']

    neg = data.loc[data['predicted']=='negative']
    Counter_neg  = Counter(neg['tweet'])
    most_words_neg = Counter_neg.most_common(5)
    dt = pd.DataFrame(most_words_neg)
    dt.columns = ['negative_tweet','count']

    neu = data.loc[data['predicted']=='neutral']
    Counter_neu  = Counter(neu['tweet'])
    most_words_neu = Counter_neu.most_common(5)
    dt_neu = pd.DataFrame(most_words_neu)
    dt_neu.columns = ['neutral_tweet','count']

    return render_template("analisis.html", pos_count=dt_pos, neg_count=dt, neu_count=dt_neu)


if __name__ == '__main__':
    app.run(debug=True)