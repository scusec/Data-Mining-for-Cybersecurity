from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf
from random import randint

import dga_model
from dga_reader import load_data, DataReader

flags = tf.flags
# data
flags.DEFINE_string('data_dir', 'dga_data',
                    'data directory. Should contain train.txt/valid.txt/test.txt with input data')
flags.DEFINE_string('train_dir', 'cv', 'training directory (models and summaries are saved there periodically)')
flags.DEFINE_string('load_model', 'cv/lr_epoch002_0.0000.model',
                    '(optional) filename of the model to load. Useful for re-starting training from a checkpoint')

# model params
flags.DEFINE_integer('rnn_size', 50, 'size of LSTM internal state')
flags.DEFINE_integer('highway_layers', 2, 'number of highway layers')
flags.DEFINE_integer('char_embed_size', 30, 'dimensionality of character embeddings')
flags.DEFINE_integer('embed_dimension', 32, 'embedding features dimensions')
flags.DEFINE_string('kernels', str([2] * 20 + [3] * 10), 'CNN kernel widths')
flags.DEFINE_string('kernel_features', str([32] * 30), 'number of features in the CNN kernel')
flags.DEFINE_integer('rnn_layers', 2, 'number of layers in the LSTM')
flags.DEFINE_float('dropout', 0.5, 'dropout. 0 = no dropout')
flags.DEFINE_integer('random_dimension', 32, 'dimension of random numbers input in generator')

# optimization
flags.DEFINE_float('learning_rate_decay', 0.5, 'learning rate decay')
flags.DEFINE_float('learning_rate', 0.001, 'starting learning rate')
flags.DEFINE_float('decay_when', 1.0, 'decay if validation perplexity does not improve by more than this much')
flags.DEFINE_float('param_init', 0.5, 'initialize parameters at')
flags.DEFINE_integer('batch_size', 64, 'number of sequences to train on in parallel')
flags.DEFINE_integer('max_epochs', 7, 'number of full passes through the training data')
flags.DEFINE_integer('max_epochs_lr', 3, 'number of epochs of training lr model')
flags.DEFINE_integer('max_epochs_gl', 5, 'number of epochs of training generator model')
flags.DEFINE_float('max_grad_norm', 5.0, 'normalize gradients at')
flags.DEFINE_integer('max_word_length', 70, 'maximum word length')
flags.DEFINE_integer('iteration', 3, 'number of iterations of lr-training before a gl-lr training')

# bookkeeping
flags.DEFINE_integer('seed', 1021, 'random number generator seed')
flags.DEFINE_integer('print_every', 200, 'how often to print current loss')
flags.DEFINE_integer('num_samples', 300, 'how many words to generate')
FLAGS = flags.FLAGS

with tf.Graph().as_default(), tf.Session() as session:
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(seed=FLAGS.seed)

    char_vocab, char_tensors, char_lens, actual_max_word_length = load_data(FLAGS.data_dir, FLAGS.max_word_length)

    # top_domain=".com"
    # with open("train.txt","w") as f:
    #     for i in range(FLAGS.num_samples):
    #         str_list="abcdefghijklmnopqrstuvwxyz0123456789-"
    #         input_len=randint(5,20)
    #         input_str=""
    #         for j in range(input_len):
    #             input_str+=str_list[randint(0,len(str_list)-1)]
    #         input_str+=top_domain
    #         input_str+='\n'
    #         f.write(input_str)
    #     f.write("a"*66+".com"+"\n")
    char_vocab_, char_tensors_, char_lens_, actual_max_word_length_ = load_data(".", FLAGS.max_word_length)
    generate_reader = DataReader(char_tensors_['train'], char_lens_['train'], FLAGS.batch_size)
    

    with tf.variable_scope("Model"):
        m = dga_model.inference_graph(
            char_vocab_size=char_vocab.size,
            char_embed_size=FLAGS.char_embed_size,
            batch_size=FLAGS.batch_size,
            num_highway_layers=FLAGS.highway_layers,
            num_rnn_layers=FLAGS.rnn_layers,
            rnn_size=FLAGS.rnn_size,
            max_word_length=actual_max_word_length,
            kernels=eval(FLAGS.kernels),
            kernel_features=eval(FLAGS.kernel_features),
            dropout=FLAGS.dropout,
            embed_dimension=FLAGS.embed_dimension)

            # we need global step only because we want to read it from the model
        global_step = tf.Variable(0, dtype=tf.int32, name='global_step')

        m.update(dga_model.decoder_graph(m.embed_output,
                                        char_vocab_size=char_vocab.size,
                                        batch_size=FLAGS.batch_size,
                                        num_highway_layers=FLAGS.highway_layers,
                                        num_rnn_layers=FLAGS.rnn_layers,
                                        rnn_size=FLAGS.rnn_size,
                                        max_word_length=actual_max_word_length,
                                        kernels=eval(FLAGS.kernels),
                                        kernel_features=eval(FLAGS.kernel_features),
                                        dropout=FLAGS.dropout,
                                        ))
        m.update(dga_model.genearator_layer(batch_size=FLAGS.batch_size,
                                            input_dimension=FLAGS.random_dimension,
                                            max_word_length=actual_max_word_length,
                                            embed_dimension=FLAGS.embed_dimension))

    saver = tf.train.Saver()
    saver.restore(session, FLAGS.load_model)
    print('Loaded model from', FLAGS.load_model, 'saved at global step', global_step.eval())
    rnn_state_g = session.run(m.initial_rnn_state_g)
    rnn_state_d = session.run(m.initial_rnn_state_d)
    np_random = np.random.RandomState(FLAGS.seed)
    with open("result.txt","w") as f_out:
        for x, y in generate_reader.iter():
        #     rnn_result=session.run(m.embed_output,{m.input:x,m.input_len_g: y})
        #     generated_dga=session.run(m.generated_dga,{m.input:x,m.input_len_g: y})
        #     for index,domain in enumerate(generated_dga):
        #         f_out.write(char_vocab.change(x[index])+"-->"+char_vocab.change(domain))
        #         f_out.write('\n')
            #for i in range(300):
            generator_input = np_random.rand(FLAGS.batch_size, FLAGS.random_dimension)
            gl_output = session.run([m.gl_output,], {m.gl_input: generator_input,})

            rnn_state_g, _, embed_output = session.run([
                        m.final_rnn_state_g,
                        m.clear_char_embedding_padding,
                        m.embed_output,
                    ], {
                        m.input: x,
                        m.input_len_g: y,
                        m.initial_rnn_state_g: rnn_state_g,
                    })

            target = np.zeros([FLAGS.batch_size])
            target[0: int(len(target) / 2)] = np.ones([int(len(target) / 2)])
            gl_output = gl_output[0]
            gl_output[0: int(len(embed_output) / 2)] = embed_output[0: int(len(embed_output) / 2)]

            gl_output = np.array(gl_output)
            # print(np.shape(gl_output))
            gl_output=np.reshape(gl_output,(FLAGS.batch_size,actual_max_word_length,FLAGS.embed_dimension))
            #print(np.shape(gl_output))
            generated_dga=session.run([m.generated_dga,],{m.decoder_input:gl_output})
            #print(np.shape(generated_dga))
            generated_dga=np.reshape(generated_dga,(FLAGS.batch_size,actual_max_word_length))
            for g in generated_dga:
                f_out.write(char_vocab.change(g)+"\n")