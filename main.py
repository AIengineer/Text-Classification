import os
import warnings
from keras.callbacks import ModelCheckpoint
import nltk
import pandas as pd
from keras import optimizers
from keras.layers import Embedding, Input, Dense, LSTM, Bidirectional, TimeDistributed, Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from nltk import tokenize
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from Model.model import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

current_path = os.getcwd()
warnings.filterwarnings('ignore')
nltk.download('punkt')

# -----------hyper-parameters of HAN_LSTM-------------------
HAN_LSTM_UNITS = 100
DROP_OUT_HAN = 0.03
HAN_LEARNING_RATE = 0.001
Epochs_HAN = 100
HAN_BATCH_SIZE = 64
# -----------hyper-parameters of HAN_LSTM-------------------

# -----------hyper-parameters of DNN------------------------
DNN_UNITS = 100
DNN_LEARNING_RATE = 0.001
DNN_EPOCHS = 100
DNN_BATCH_SIZE = 64
# -----------hyper-parameters of DNN------------------------

# -----------hyper-parameters of CNN------------------------
CNN_filters = 6
CNN_FULLY_CONNECTED_UNITS = 100
CNN_LEARNING_RATE = 0.001
CNN_EPOCHS = 100
CNN_BATCH_SIZE = 64
# -----------hyper-parameters of CNN------------------------

# -----------read data--------------------------------------
Product_Behavioral_Features = pd.read_excel('./data/Resulting Features_Product_Centric_Behavioral_Features.xlsx')
Product_Textual_Features = pd.read_excel('./data/Resulting_Features_Product_Centric_Textual_Features.xlsx')
Review_Behavioral_Features = pd.read_excel('./data/Review_Centric_Behavioral_Features.xlsx')
Review_Textual_Features = pd.read_excel('./data/Review_Centric_Textual_Features.xlsx')
Reviewer_Textual_Features = pd.read_excel('./data/Reviewer_Centric_Textual_Features.xlsx')
Reviewer_Behavioral_Features = pd.read_excel('./data/Reviewer_Centric_Behavioral_Features.xlsx')
print('Product_Behavioral_Features length : {} and it`s shape: {} \n'.format(len(Product_Behavioral_Features),
                                                                             Product_Behavioral_Features.shape))
print('Product_Textual_Features length    : {} and it`s shape: {} \n'.format(len(Product_Textual_Features),
                                                                             Product_Textual_Features.shape))
print('Review_Behavioral_Features length  : {} and it`s shape: {} \n'.format(len(Review_Behavioral_Features),
                                                                             Review_Behavioral_Features.shape))
print('Review_Textual_Features length     : {} and it`s shape: {} \n'.format(len(Review_Textual_Features),
                                                                             Review_Textual_Features.shape))
print('Reviewer_Textual_Features length   : {} and it`s shape: {} \n'.format(len(Reviewer_Textual_Features),
                                                                             Reviewer_Textual_Features.shape))
print('Reviewer_Behavioral_Features length: {} and it`s shape: {} \n'.format(len(Reviewer_Behavioral_Features),
                                                                             Reviewer_Behavioral_Features.shape))

product_feature = Product_Behavioral_Features.join(Product_Textual_Features, how='outer')
reviewer_feature = Reviewer_Behavioral_Features.join(Reviewer_Textual_Features, how='outer')
review_feature = Review_Behavioral_Features.join(Review_Textual_Features, how='outer')

Labels_for_products = pd.read_excel("./data/Labels_for_products.xlsx")
Labels_for_reviewers = pd.read_excel("./data/Labels_for_reviewers.xlsx")
Labels_for_reviews = pd.read_excel("./data/Labels_for_reviews.xlsx")
Metadata_Sortedby_Product_wise = pd.read_excel("./data/Metadata (Sortedby_Product_wise).xlsx")
Metadata_Sortedby_Reviewer_wise = pd.read_excel("./data/Metadata (Sortedby_Reviewer_wise).xlsx")
ReviewContent_Sortedby_Product_wise = pd.read_excel("./data/ReviewContent (Sortedby_Product_wise).xlsx")
ReviewContent_Sortedby_Reviewer_wise = pd.read_excel("./data/ReviewContent (Sortedby_Reviewer_wise).xlsx")

print('Labels_for_products length                  : {} and it`s shape: {} \n'.format(len(Labels_for_products),
                                                                                      Labels_for_products.shape))
print('Labels_for_reviewers length                 : {} and it`s shape: {} \n'.format(len(Labels_for_reviewers),
                                                                                      Labels_for_reviewers.shape))
print('Labels_for_reviews length                   : {} and it`s shape: {} \n'.format(len(Labels_for_reviews),
                                                                                      Labels_for_reviews.shape))
print(
    'Metadata_Sortedby_Product_wise length     : {} and it`s shape: {} \n'.format(len(Metadata_Sortedby_Product_wise),
                                                                                  Metadata_Sortedby_Product_wise.shape))
print('Metadata_Sortedby_Reviewer_wise length      : {} and it`s shape: {} \n'.format(
    len(Metadata_Sortedby_Reviewer_wise), Metadata_Sortedby_Reviewer_wise.shape))
print('ReviewContent_Sortedby_Product_wise length  : {} and it`s shape: {} \n'.format(
    len(ReviewContent_Sortedby_Product_wise), ReviewContent_Sortedby_Product_wise.shape))
print('ReviewContent_Sortedby_Reviewer_wise length : {} and it`s shape: {} \n'.format(
    len(ReviewContent_Sortedby_Reviewer_wise), ReviewContent_Sortedby_Reviewer_wise.shape))

Labels_for_reviews.rename(
    columns={'Reviewer_id': 'Reviewer_id_from_Labels_for_reviews', 'Label': 'Label_Labels_for_reviews'}, inplace=True)
ReviewContent_Sortedby_Product_wise.rename(
    columns={'Reviewer_id': 'Reviewer_id_from_ReviewContent_Sortedby_Product_wise',
             'Product_id': 'Product_id_from_ReviewContent_Sortedby_Product_wise',
             'Date': 'Date_from_ReviewContent_Sortedby_Product_wise',
             'Text': 'Text_from_ReviewContent_Sortedby_Product_wise'}, inplace=True)
Metadata_Sortedby_Product_wise.rename(columns={'Reviewer_id': 'Reviewer_id_fromMeta_sortby_productwise',
                                               'Product_id': 'Product_id_fromMeta_sortby_productwise',
                                               'Label': 'Label_from_meta_product_wise',
                                               'Rating': 'Rating_fromMeta_sortby_productwise',
                                               'Date': 'Date_fromMeta_sortby_productwise'}, inplace=True)
#
review_feature_with_data = review_feature.join(ReviewContent_Sortedby_Product_wise[
                                                   ['Reviewer_id_from_ReviewContent_Sortedby_Product_wise',
                                                    'Product_id_from_ReviewContent_Sortedby_Product_wise',
                                                    'Date_from_ReviewContent_Sortedby_Product_wise',
                                                    'Text_from_ReviewContent_Sortedby_Product_wise']], how='outer')
review_feature_with_data = review_feature_with_data.join(Metadata_Sortedby_Product_wise, how='outer')
review_feature_with_data = review_feature_with_data.join(Labels_for_reviews, how='outer')

product_final = product_feature.join(Labels_for_products)
review_final = review_feature.join(Labels_for_reviews)
reviewer_final = reviewer_feature.join(Labels_for_reviewers)

feature_with_id = review_feature_with_data
reviewr_id_transform = reviewer_final
product_id_transform = product_final

reviewr_id_transform = reviewr_id_transform.rename(columns={'Reviewer_id': 'Reviewer_id_from_Labels_for_reviews'})
product_id_transform = product_id_transform.rename(columns={'Product_id': 'Product_id_fromMeta_sortby_productwise'})

feature_with_reviewr = pd.merge(feature_with_id, reviewr_id_transform, on=['Reviewer_id_from_Labels_for_reviews'],
                                how='left')
feature_with_reviewr2 = feature_with_reviewr.add_suffix("_r")
product_id_transform2 = product_id_transform.add_suffix("_p")
multi_view_features = pd.merge(feature_with_reviewr2, product_id_transform2,
                               left_on='Product_id_fromMeta_sortby_productwise_r',
                               right_on='Product_id_fromMeta_sortby_productwise_p',
                               how='left').replace(to_replace={-1: 0, 1: 1})

data = review_feature_with_data

data['Text_from_ReviewContent_Sortedby_Product_wise'].replace(to_replace={np.nan: ""}, inplace=True)
new_data = data[['Text_from_ReviewContent_Sortedby_Product_wise', 'Label_Labels_for_reviews']].rename(
    columns={'Text_from_ReviewContent_Sortedby_Product_wise': "text", 'Label_Labels_for_reviews': "category"})
new_data = new_data.replace(to_replace={-1: 0, 1: 1})
new_data.to_csv("yelp_text.csv", index=False)
# ----------- read data done--------------------------------------

max_features = 200000
max_senten_len = 40
max_senten_num = 6
embed_size = 300
VALIDATION_SPLIT = 0.05

df = new_data.reset_index(drop=True)

categories = df['category']
text = df['text']
paras = []
labels = []
texts = []

sent_lens = []
sent_nums = []
for idx in range(df.text.shape[0]):
    text = clean_str(df.text[idx])
    texts.append(text)
    sentences = tokenize.sent_tokenize(text)
    sent_nums.append(len(sentences))
    for sent in sentences:
        sent_lens.append(len(text_to_word_sequence(sent)))
    paras.append(sentences)

tokenizer = Tokenizer(num_words=max_features, oov_token=True)
tokenizer.fit_on_texts(texts)

data = np.zeros((len(texts), max_senten_num, max_senten_len), dtype='int32')
for i, sentences in enumerate(paras):
    for j, sent in enumerate(sentences):
        if j < max_senten_num:
            wordTokens = text_to_word_sequence(sent)
            k = 0
            for _, word in enumerate(wordTokens):
                try:
                    if k < max_senten_len and tokenizer.word_index[word] < max_features:
                        data[i, j, k] = tokenizer.word_index[word]
                        k = k + 1
                except:
                    print(word)
                    pass

word_index = tokenizer.word_index

labels = categories

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]

nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

REG_PARAM = 1e-13
l2_reg = regularizers.l2(12)

GLOVE_DIR = "./data/glove.6B.300d.txt"
embeddings_index = {}
f = open(GLOVE_DIR, encoding="utf8")
for line in f:
    try:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    except:
        pass
f.close()

embedding_matrix = np.zeros((len(word_index) + 1, embed_size))
absent_words = 0
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    else:
        absent_words += 1

# ----------------- emdedding layer---------------------------------------------------
embedding_layer = Embedding(len(word_index) + 1, embed_size, weights=[embedding_matrix], input_length=max_senten_len,
                            trainable=True)
word_input = Input(shape=(max_senten_len,), dtype='float32')
word_sequences = embedding_layer(word_input)
word_lstm = Bidirectional(LSTM(HAN_LSTM_UNITS, return_sequences=True, kernel_regularizer=l2_reg))(word_sequences)
word_dense = TimeDistributed(Dense(4, kernel_regularizer=l2_reg))(word_lstm)
word_att = AttentionWithContext()(word_dense)
wordEncoder = Model(word_input, word_att)

sent_input = Input(shape=(max_senten_num, max_senten_len), dtype='float32')
sent_encoder = TimeDistributed(wordEncoder)(sent_input)
sent_lstm = Bidirectional(LSTM(HAN_LSTM_UNITS, return_sequences=True, kernel_regularizer=l2_reg))(sent_encoder)
sent_dense = TimeDistributed(Dense(4, kernel_regularizer=l2_reg))(sent_lstm)
sent_att = Dropout(DROP_OUT_HAN)(AttentionWithContext()(sent_dense))
preds = Dense(1, activation='sigmoid')(sent_att)
model = Model(sent_input, preds)
optimizer = optimizers.Adam()
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
checkpoint = ModelCheckpoint(os.path.join(current_path, 'LSTM', 'best_LSTM_model.h5'), verbose=0, save_weights_only=True,
                             monitor='val_loss', save_best_only=True, mode='auto')
callback = keras.callbacks.TensorBoard(log_dir=os.path.join(current_path, 'LSTM'),
                                       histogram_freq=0, write_graph=False, write_images=False)
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    verbose=1,
    patience=20,
    mode='max',
    restore_best_weights=True)

history = model.fit(x_train, y_train, epochs=Epochs_HAN, batch_size=HAN_BATCH_SIZE, validation_split=0.2,
                    callbacks=[checkpoint, callback, early_stopping])
model.load_weights(os.path.join(current_path, 'LSTM', 'best_LSTM_model.h5'))
y_pred = np.around(model.predict(x_val))

print(classification_report(y_val, y_pred))

cm = confusion_matrix(y_val, y_pred)
plot_confusion_matrix(cm, classes=['0', '1'])
print("============== LSTM end! ============")

product_feature_s = product_feature.add_suffix("_p")
x_product = product_feature_s.values
y_product = Labels_for_products.Label.replace(to_replace={-1: 0}).values
X_train_product, X_test_product, y_train_product, y_test_product = train_test_split(x_product, y_product,
                                                                                    train_size=0.8, test_size=0.2,
                                                                                    stratify=y_product)

reviewer_feature_s = reviewer_feature.add_suffix("_r")

x_reviewer = reviewer_feature_s.values
y_reviewer = Labels_for_reviewers.Label.replace(to_replace={-1: 0}).values
X_train_reviewer, X_test_reviewer, y_train_reviewer, y_test_reviewer = train_test_split(x_reviewer, y_reviewer,
                                                                                        train_size=0.8, test_size=0.2,
                                                                                        stratify=y_reviewer)

DNNmodel = Sequential()
DNNmodel.add(Dense(DNN_UNITS, input_dim=22, activation='relu'))
DNNmodel.add(Dense(1, activation='sigmoid'))
optimizer = Adam(DNN_LEARNING_RATE)

DNNmodel.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
checkpoint = ModelCheckpoint(os.path.join(current_path, 'DNN', 'best_DNN_model.h5'), verbose=0, save_weights_only=True,
                             monitor='val_loss', save_best_only=True, mode='auto')
callback = keras.callbacks.TensorBoard(log_dir=os.path.join(current_path, 'DNN'), histogram_freq=0, write_graph=False, write_images=False)
DNNmodel.fit(X_train_product, y_train_product, epochs=DNN_EPOCHS, batch_size=DNN_BATCH_SIZE, validation_split=0.2,
             callbacks=[early_stopping, callback, checkpoint])
DNNmodel.load_weights(os.path.join(current_path, 'DNN', 'best_DNN_model.h5'))
_, Accuracy = DNNmodel.evaluate(X_train_product, y_train_product)
print("============== DNN ============")

print('Accuracy: %.2f' % (Accuracy * 100))

target_names = ['class 0', 'class 1']
y_pred = DNNmodel.predict_classes(X_test_product)
print(classification_report(y_test_product, y_pred, target_names=target_names))
print("accuracy = ", accuracy_score(y_test_product, y_pred))
cm = confusion_matrix(y_test_product, y_pred)
plot_confusion_matrix(cm, classes=['0', '1'])
print("============== DNN end ============")

CNNmodel = Sequential()
CNNmodel.add(Conv1D(filters=CNN_filters, kernel_size=3, activation='relu',
                    input_shape=(23, 1)))  # 577 is the number of features
CNNmodel.add(Flatten())
CNNmodel.add(Dense(CNN_FULLY_CONNECTED_UNITS, activation='relu'))
CNNmodel.add(Dense(1, activation='sigmoid'))
X_train_reviewer = X_train_reviewer.reshape(len(X_train_reviewer), X_train_reviewer.shape[1], 1)
X_test_reviewer = X_test_reviewer.reshape(len(X_test_reviewer), X_test_reviewer.shape[1], 1)

optimizer = Adam(CNN_LEARNING_RATE)
CNNmodel.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
checkpoint = ModelCheckpoint(os.path.join(current_path, 'CNN', 'best_DNN_model.h5'), verbose=0, save_weights_only=True,
                             monitor='val_loss', save_best_only=True, mode='auto')
callback = keras.callbacks.TensorBoard(log_dir=os.path.join(current_path, 'CNN'),
                                       histogram_freq=0, write_graph=False, write_images=False)
CNNmodel.fit(X_train_reviewer, y_train_reviewer, epochs=CNN_EPOCHS, batch_size=CNN_BATCH_SIZE, validation_split=0.2,
             callbacks=[early_stopping, callback, checkpoint])
CNNmodel.load_weights(os.path.join(current_path, 'CNN', 'best_DNN_model.h5'))
_, Accuracy = CNNmodel.evaluate(X_train_reviewer, y_train_reviewer)

print("============== CNN ============")
print('Accuracy: %.2f' % (Accuracy * 100))

arget_names = ['class 0', 'class 1']
y_pred = CNNmodel.predict_classes(X_test_reviewer)
print(classification_report(y_test_reviewer, y_pred, target_names=target_names))
print("accuracy =", accuracy_score(y_test_reviewer, y_pred))

cm = confusion_matrix(y_test_reviewer, y_pred)
plot_confusion_matrix(cm, classes=['0', '1'])
print("============== CNN end============")


def ensemble_predictions(DNNmodel, CNNmodel, HANmodel, df):
    # make predictions
    testX_reviewer = df[reviewer_feature_s.columns].values
    testX_product = df[product_feature_s.columns].values
    tesX_review = encode_text(df.Text_from_ReviewContent_Sortedby_Product_wise_r.values)
    CNN_yhats = CNNmodel.predict(testX_reviewer.reshape((len(testX_reviewer), testX_reviewer.shape[1], 1)))
    DNN_yhats = DNNmodel.predict(testX_product.reshape((len(testX_product), testX_product.shape[1])))
    HAN_yhats = HANmodel.predict(tesX_review)
    yhats = np.array([DNN_yhats, CNN_yhats, HAN_yhats]).squeeze()
    Soft = np.sum(yhats, axis=0)
    super_threshold_indices = Soft > 1.5
    Soft[super_threshold_indices] = 0
    Soft[np.flip(super_threshold_indices)] = 1
    Hard = np.around(yhats.astype(np.float32)).astype(np.float32)
    Hard = np.sum(Hard, axis=0)
    print("Hard unique = ", set(Hard))
    # Hard = [1 if a_ > 1.5 else 0 for a_ in Hard]
    super_threshold_indices = Hard > 1.5
    Hard[super_threshold_indices] = 0
    Hard[np.flip(super_threshold_indices)] = 1
    print("Soft unique = ", set(Soft))
    return Soft, Hard


def encode_text(encoded_text):
    # print(encoded_text)
    paras = []
    labels = []
    texts = []

    sent_lens = []
    sent_nums = []
    for idx in tqdm(range(len(encoded_text)), position=0, leave=True):
        # print(idx)
        text = clean_str(encoded_text[idx])
        texts.append(text)
        sentences = tokenize.sent_tokenize(text)
        sent_nums.append(len(sentences))
        for sent in sentences:
            sent_lens.append(len(text_to_word_sequence(sent)))
        paras.append(sentences)
    data = np.zeros((len(texts), max_senten_num, max_senten_len), dtype='int32')
    # print("data = ", data.shape)
    for i, sentences in enumerate(paras):
        for j, sent in enumerate(sentences):
            if j < max_senten_num:
                wordTokens = text_to_word_sequence(sent)
                k = 0
                for _, word in enumerate(wordTokens):
                    try:
                        if k < max_senten_len and tokenizer.word_index[word] < max_features:
                            data[i, j, k] = tokenizer.word_index[word]
                            k = k + 1
                    except:
                        print(word)
                        pass
    return data


Soft_pred, Hard_pred = ensemble_predictions(DNNmodel, CNNmodel, model, multi_view_features.iloc[:100000])
print("===================== Voting =======================")
print("Soft voting = ", accuracy_score(multi_view_features.Label_Labels_for_reviews_r[:100000], Soft_pred))
print("Hard voting = ", accuracy_score(multi_view_features.Label_Labels_for_reviews_r[:100000], Hard_pred))

cm = confusion_matrix(multi_view_features.Label_Labels_for_reviews_r[:100000], Soft_pred)
plot_confusion_matrix(cm, classes=['0', '1'])