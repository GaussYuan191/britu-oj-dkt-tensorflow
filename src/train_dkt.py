import argparse
import sys
import time
# import sysdata
from src.TensorFlowDKT import *
from src.data_process import *
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score

from src.data_process import getDataByMysql, split_dataset



def run(u_id):

    # process data
    seqs_by_student, num_skills = getDataByMysql()
    train_seqs = seqs_by_student
    test_seqs = list(u_id)
    batch_size = 10
    train_generator = DataGenerator(train_seqs, batch_size=batch_size, num_skills=num_skills)
    test_generator = DataGenerator(test_seqs, batch_size=batch_size, num_skills=num_skills)

    # config and create model
    config = {"hidden_neurons": [200],
              "batch_size": batch_size,
              "keep_prob": 0.6,     #保持概率
              "num_skills": num_skills,
              "input_size": num_skills * 2}
    model = TensorFlowDKT(config)

    # create session and init all variables
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()  # 生成saver
    lr = 0.4
    lr_decay = 0.92
    ability = 0
    # run epoch
    for epoch in range(10):
        # train
        model.assign_lr(sess, lr * lr_decay ** epoch)
        overall_loss = 0
        train_generator.shuffle()
        st = time.time()
        while not train_generator.end:
            input_x, target_id, target_correctness, seqs_len, max_len = train_generator.next_batch()
            overall_loss += model.step(sess, input_x, target_id, target_correctness, seqs_len, is_train=True)
            print ("\r idx:{0}, overall_loss:{1}, time spent:{2}s".format(train_generator.pos, overall_loss,
                                                                         time.time() - st))
            sys.stdout.flush()

        # test 测试
        test_generator.reset()
        preds, binary_preds, targets = list(), list(), list()
        while not test_generator.end:
            input_x, target_id, target_correctness, seqs_len, max_len = test_generator.next_batch()
            binary_pred, pred, _ = model.step(sess, input_x, target_id, target_correctness, seqs_len, is_train=False)
            for seq_idx, seq_len in enumerate(seqs_len):
                preds.append(pred[seq_idx, 0:seq_len])
                binary_preds.append(binary_pred[seq_idx, 0:seq_len])
                targets.append(target_correctness[seq_idx, 0:seq_len])
        # compute metrics
        preds = np.concatenate(preds)
        binary_preds = np.concatenate(binary_preds)
        targets = np.concatenate(targets)
        auc_value = roc_auc_score(targets, preds)
        accuracy = accuracy_score(targets, binary_preds)
        precision, recall, f_score, _ = precision_recall_fscore_support(targets, binary_preds)
        print("\n f_score={}".format(f_score))
        print("\n auc={0}, accuracy={1}, precision={2}, recall={3}".format(auc_value, accuracy, precision, recall))
        saver.save(sess, "../model/dkt-model")
        ability += f_score


    result = list(ability/10)
    return (result[0] + result[1]) / 2
if __name__ == "__main__":

    print(run())
