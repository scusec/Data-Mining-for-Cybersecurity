#encoding=utf-8

# @Time:2019/9/22 17:50
# @Author:wyx
# @File:test.py

import tensorflow as tf
import numpy as np

glove_vectors_file = "glove.6B.50d.txt"
snli_dev_file = "snli_1.0_dev.txt"

max_hypothesis_length, max_evidence_length = 30, 30
batch_size, vector_size, hidden_size = 128, 50, 64
lstm_size = hidden_size
weight_decay = 0.0001
learning_rate = 1
input_p, output_p = 0.5, 0.5
training_iterations_count = 100000
display_step = 10

glove_wordmap = {}

def sentence2sequence(sentence):
    """

    - Turns an input sentence into an (n,d) matrix,
        where n is the number of tokens in the sentence
        and d is the number of dimensions each word vector has.

      Tensorflow doesn't need to be used here, as simply
      turning the sentence into a sequence based off our
      mapping does not need the computational power that
      Tensorflow provides. Normal Python suffices for this task.
    """
    tokens = sentence.lower().split(" ")
    # print("Tokens: ", tokens)
    rows = []
    words = []
    # Greedy search for tokens
    for token in tokens:
        i = len(token)
        # print("Token: ", token, "Length: ", i)
        while len(token) > 0 and i > 0:
            word = token[:i]
            # print("Word: ", word)
            if word in glove_wordmap:
                rows.append(glove_wordmap[word])
                words.append(word)
                token = token[i:]
                i = len(token)
            else:
                i = i - 1
    return rows, words

# The scores are relative certainties for how likely the output matches
#   a certain entailment:
#     0: Positive entailment
#     1: Neutral entailment
#     2: Negative entailment

def score_setup(row):
    convert_dict = {
      'entailment': 0,
      'neutral': 1,
      'contradiction': 2
    }
    score = np.zeros((3,))
    for x in range(1,6):
        tag = row["label"+str(x)]
        if tag in convert_dict: score[convert_dict[tag]] += 1
    return score / (1.0*np.sum(score))

def fit_to_size(matrix, shape):
    res = np.zeros(shape)
    slices = [slice(0,min(dim,shape[e])) for e, dim in enumerate(matrix.shape)]
    res[slices] = matrix[slices]
    return res


def split_data_into_scores():
    import csv
    with open("snli_1.0_dev.txt", "r") as data:
        train = csv.DictReader(data, delimiter='\t')
        evi_sentences = []
        hyp_sentences = []
        labels = []
        scores = []
        for row in train:
            hyp_sentences.append(np.vstack(
                sentence2sequence(row["sentence1"].lower())[0]))
            evi_sentences.append(np.vstack(
                sentence2sequence(row["sentence2"].lower())[0]))
            labels.append(row["gold_label"])
            scores.append(score_setup(row))

        hyp_sentences = np.stack([fit_to_size(x, (max_hypothesis_length, vector_size))
                                  for x in hyp_sentences])
        evi_sentences = np.stack([fit_to_size(x, (max_evidence_length, vector_size))
                                  for x in evi_sentences])

        return (hyp_sentences, evi_sentences), labels, np.array(scores)

with open(glove_vectors_file, "r", encoding='UTF-8') as glove:
    for line in glove:
        name, vector = tuple(line.split(" ", 1))
        glove_wordmap[name] = np.fromstring(vector, sep=" ")


tf.reset_default_graph()

data_feature_list, correct_values, correct_scores = split_data_into_scores()

l_h, l_e = max_hypothesis_length, max_evidence_length
N, D, H = batch_size, vector_size, hidden_size
l_seq = l_h + l_e



hyp = tf.placeholder(tf.float32, [N, l_h, D], 'hypothesis')
evi = tf.placeholder(tf.float32, [N, l_e, D], 'evidence')
y = tf.placeholder(tf.float32, [N, 3], 'label')

x = tf.concat([hyp, evi], 1) # N, (Lh+Le), d
# Permuting batch_size and n_steps
x = tf.transpose(x, [1, 0, 2]) # (Le+Lh), N, d
# Reshaping to (n_steps*batch_size, n_input)
x = tf.reshape(x, [-1, vector_size]) # (Le+Lh)*N, d
# Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
x = tf.split(x, l_seq,)

lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
lstm_drop =  tf.contrib.rnn.DropoutWrapper(lstm, input_p, output_p)
lstm_back = tf.contrib.rnn.BasicLSTMCell(lstm_size)
lstm_drop_back = tf.contrib.rnn.DropoutWrapper(lstm_back, input_p, output_p)
rnn_outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm, lstm_back,
                                                            x, dtype=tf.float32)

fc_initializer = tf.random_normal_initializer(stddev=0.1)
fc_weight = tf.get_variable('fc_weight', [2*hidden_size, 3],
                            initializer = fc_initializer)
fc_bias = tf.get_variable('bias', [3])
tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                     tf.nn.l2_loss(fc_weight))

classification_scores = tf.matmul(rnn_outputs[-1], fc_weight) + fc_bias



with tf.variable_scope('Accuracy'):
    predicts = tf.cast(tf.argmax(classification_scores, 1), 'int32')
    y_label = tf.cast(tf.argmax(y, 1), 'int32')
    corrects = tf.equal(predicts, y_label)
    num_corrects = tf.reduce_sum(tf.cast(corrects, tf.float32))
    accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))

with tf.variable_scope("loss"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits = classification_scores, labels = y)
    loss = tf.reduce_mean(cross_entropy)
    total_loss = loss + weight_decay * tf.add_n(
        tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

optimizer = tf.train.GradientDescentOptimizer(learning_rate)

opt_op = optimizer.minimize(total_loss)

# Initialize variables
init = tf.global_variables_initializer()

# Use TQDM if installed
tqdm_installed = False
try:
    from tqdm import tqdm

    tqdm_installed = True
except:
    pass

# Launch the Tensorflow session
sess = tf.Session()
sess.run(init)

# training_iterations_count: The number of data pieces to train on in total
# batch_size: The number of data pieces per batch
training_iterations = range(0, training_iterations_count, batch_size)
if tqdm_installed:
    # Add a progress bar if TQDM is installed
    training_iterations = tqdm(training_iterations)

for i in training_iterations:

    # Select indices for a random data subset
    batch = np.random.randint(data_feature_list[0].shape[0], size=batch_size)

    # Use the selected subset indices to initialize the graph's
    #   placeholder values
    hyps, evis, ys = (data_feature_list[0][batch, :],
                      data_feature_list[1][batch, :],
                      correct_scores[batch])

    # Run the optimization with these initialized values
    sess.run([opt_op], feed_dict={hyp: hyps, evi: evis, y: ys})
    # display_step: how often the accuracy and loss should
    #   be tested and displayed.
    if (i / batch_size) % display_step == 0:
        # Calculate batch accuracy
        acc = sess.run(accuracy, feed_dict={hyp: hyps, evi: evis, y: ys})
        # Calculate batch loss
        tmp_loss = sess.run(loss, feed_dict={hyp: hyps, evi: evis, y: ys})
        # Display results
        print("Iter " + str(i / batch_size) + ", Minibatch Loss= " + \
              "{:.6f}".format(tmp_loss) + ", Training Accuracy= " + \
              "{:.5f}".format(acc))

evidences = ["Maurita and Jade both were at the scene of the car crash."]

hypotheses = ["Multiple people saw the accident."]

sentence1 = [fit_to_size(np.vstack(sentence2sequence(evidence)[0]),
                         (30, 50)) for evidence in evidences]

sentence2 = [fit_to_size(np.vstack(sentence2sequence(hypothesis)[0]),
                         (30,50)) for hypothesis in hypotheses]

prediction = sess.run(classification_scores, feed_dict={hyp: (sentence1 * N),
                                                        evi: (sentence2 * N),
                                                        y: [[0,0,0]]*N})
print(["Positive", "Neutral", "Negative"][np.argmax(prediction[0])]+
      " entailment")
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.close()