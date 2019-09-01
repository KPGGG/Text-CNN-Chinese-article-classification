import tensorflow as tf
import  tensorflow.contrib.slim as slim



class TextCNN:
    def __init__(self, config, vocab_size):
        #超参数
        self.config = config
        self.vocab_size = vocab_size

        self.num_classes = config["num_classes"]
        self.sequence_length = config["sequence_length"]
        self.embedding_size = config["embedding_size"]
        self.filter_sizes = config["filter_sizes"]
        self.num_filters = config["num_filters"]
        self.L2_reg_lambda = config["L2_reg_lambda"]


        #占位符
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name = 'input_x')
        self.input_y = tf.placeholder(tf.float32, [None], name = 'input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name = 'dropout_keep_prob')

        # to one_hot
        self.input_y_one_hot = tf.one_hot(self.input_y, depth=self.num_classes)

        #
        self.model_structure()
        #
        self.init_saver()

    def model_structure(self):


        # 词嵌入
        with tf.variable_scope('Embedding'):
            embed = tf.contrib.layers.embed_sequence(self.input_x, vocab_size=self.vocab_size,
                                                     embed_dim = self.embedding_size)
            self.embedded_chars_expanded = tf.expand_dims(embed, -1)

        # 定义多通道卷积与最大池化
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            conv = slim.conv2d(self.embedded_chars_expanded, num_outputs=self.num_filters,
                               kernel_size=[filter_size, self.embedding_size], stride= 1, padding="VALID",
                               activation_fn=tf.nn.leaky_relu, scope="conv{}".format(filter_size))
            pooled = slim.max_pool2d(conv, [self.sequence_length - filter_size + 1, 1], padding="VALID")
            pooled_outputs.append(pooled)

        num_filters_total = self.num_filters*len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # 计算L2_loss
        l2_loss = tf.constant(0.0)
        with tf.name_scope("output"):
            self.scores = slim.fully_connected(self.h_drop, self.num_classes,
                                               activation_fn=None,
                                               scope="fully_connected")
            for tf_var in tf.trainable_variables():
                if ("fully_connected" in tf_var.name ):
                    l2_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))
                    print("tf_var", tf_var)

            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # 计算交叉熵
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y_one_hot)
            self.loss = tf.reduce_mean(losses) + self.L2_reg_lambda*l2_loss

        # 计算准确率
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions,
                                           tf.argmax(self.input_y_one_hot, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        # 定义优化器
        #self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        train_able_var = tf.trainable_variables()
        gradient = tf.gradients(self.loss, train_able_var)

        # 梯度截断
        clip_gradient, _ = tf.clip_by_global_norm(gradient, 5)
        self.train_op = optimizer.apply_gradients(zip(clip_gradient,train_able_var))



    def init_saver(self):
        self.saver = tf.train.Saver(tf.global_variables)

    def train(self, sess,  batch, dropout_prob):
        feed_dict ={
            self.input_x:batch['file'],
            self.input_y:batch['labels'],
            self.dropout_keep_prob:dropout_prob
        }
        _, loss, predictions, accuracy = sess.run([self.train_op, self.loss, self.predictions, self.accuracy], feed_dict=[feed_dict])
        return loss, predictions, accuracy

    def eval(self, sess, batch):
        feed_dict = {
            self.input_x: batch['file'],
            self.input_y: batch['labels'],
            self.dropout_keep_prob: 1.0
        }
        loss, predictions, accuracy= sess.run([self.loss, self.predictions, self.accuracy], feed_dict=feed_dict)
        return loss, predictions, accuracy

    def infer(self, sess, batch):
        feed_dict = {
            self.input_x: batch['file'],
            self.input_y: batch['labels'],
            self.dropout_keep_prob: 1.0
        }
        loss, predictions = sess.run([self.loss, self.predictions], feed_dict=feed_dict)
        return loss, predictions

    def build_mode(self):
        """
        定义函数构建模型
        :return:
        """
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step= self.global_step)

        #生成摘要
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summart = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparisty_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_summaries)
                grad_summaries.append(sparisty_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)
        loss_summary = tf.summary.scalar('loss', self.loss)
        acc_summary = tf.summary.scalar('accuracy', self.accuracy)

        self.train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
