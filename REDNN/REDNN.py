import numpy as np
import tensorflow as tf
import string
import time

file = open("result.txt","w")
file2 = open("result2.txt","w")
file3 = open("result3.txt","w")
file4 = open("result4.txt","w")
file5 = open("result5.txt","w")

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)


def bias_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

# 데이터 파싱 구간, 현재는 3개의 label 외에는 outlier로 취급하였음
input_Data = np.loadtxt(open("./total.csv","rb"), delimiter=",", skiprows= 0)
input_Data[:,12288] = 0
input_Data[:,12289] = 0
input_Data[:,12292] = 0
input_Data[:,12293] = 0
input_Data[:,12294] = 0
input_Data[:,12295] = 0
input_Data[:,12296] = 0
input_Data[:,12297] = 0
input_Data[:,12298] = 0
input_Data[:,12300] = 0
input_Data[:,12301] = 0
input_Data[:,12302] = 0
input_Data[:,12303] = 0

for i in range(1194):
	if np.sum(input_Data[i,12288:]) == 0:
		input_Data[i, 12288] = 1
	if np.sum(input_Data[i,12290]) == 1:
		input_Data[i, 12290] = 0
		input_Data[i, 12289] = 1
	if np.sum(input_Data[i,12291]) == 1:
		input_Data[i, 12291] = 0
		input_Data[i, 12290] = 1
	if np.sum(input_Data[i,12299]) == 1:
		input_Data[i, 12299] = 0
		input_Data[i, 12291] = 1

# 테스트 데이터셋을 나누는 과정
print(input_Data.shape)
test_Data_index = [i for i in range(1194)]
np.random.shuffle(test_Data_index)
test_Data_index = test_Data_index[0:100]
print(test_Data_index)

test_Data = input_Data[test_Data_index]
input_Data = np.delete(input_Data, test_Data_index, axis=0)

print(input_Data.shape)

# 트레이닝 데이터의 augmentation을 진행하는 부분
input_Data = np.tile(input_Data,[10,1])
input_Data[:,0:12288] = input_Data[:,0:12288] * np.random.normal(1,0.01,(10940,12288))
np.random.shuffle(input_Data)
print(input_Data)

# 네트워크를 정의하는 부분
x = tf.placeholder(tf.float32, shape = [None, 12288])
y_ = tf.placeholder(tf.float32, shape = [None, 4])
keep_prob = tf.placeholder(tf.float32)
phase = tf.placeholder(tf.bool, name = 'phase')

W_1 = weight_variable([12288,512])
b_1= bias_variable([512])
y_1 = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.batch_norm((tf.matmul(x,W_1) + b_1),is_training=phase)),keep_prob)

W_2 = weight_variable([512,256])
b_2= bias_variable([256])
y_2 = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.batch_norm((tf.matmul(y_1,W_2) + b_2),is_training=phase)),keep_prob)

W_3 = weight_variable([256,128])
b_3= bias_variable([128])
y_3 = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.batch_norm((tf.matmul(y_2,W_3) + b_3),is_training=phase)),keep_prob)

W_4 = weight_variable([128,64])
b_4= bias_variable([64])
y_4 = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.batch_norm((tf.matmul(y_3,W_4) + b_4),is_training=phase)),keep_prob)

W_5 = weight_variable([64,32])
b_5 = bias_variable([32])
y_5 = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.batch_norm((tf.matmul(y_4,W_5) + b_5),is_training=phase)),keep_prob)

W_6 = weight_variable([32,4])
b_6 = bias_variable([4])
y = tf.matmul(y_5,W_6) + b_6
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
	train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

yt = tf.argmax(y,1)
yt_ = tf.argmax(y_,1)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 학습이 진행되어지는 코드
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for _ in range(1000000) :
		temp_index = [i for i in range(10940)]
		np.random.shuffle(temp_index)
		minbatch_index = temp_index[0:1000]
		train_step.run(feed_dict = {x: input_Data[minbatch_index,0:12288], y_: input_Data[minbatch_index,12288:12292], keep_prob: 0.9, phase: 1})
		if _ % 10 == 0:	
			print("test accuracy")
			print(accuracy.eval(feed_dict={x:test_Data[:,0:12288], y_: test_Data[:,12288:12292], keep_prob: 1, phase: 0}))
			file.write(str(accuracy.eval(feed_dict={x:test_Data[:,0:12288], y_: test_Data[:,12288:12292], keep_prob: 1, phase: 0})))
			file.write("\n")
			print("train accuracy")
			print(accuracy.eval(feed_dict={x: input_Data[minbatch_index,0:12288], y_: input_Data[minbatch_index,12288:12292], keep_prob: 1, phase: 0}))
			file2.write(str(accuracy.eval(feed_dict={x: input_Data[minbatch_index,0:12288], y_: input_Data[minbatch_index,12288:12292], keep_prob: 1, phase: 0})))
			file2.write("\n")		
			print("loss")
			print(cross_entropy.eval(feed_dict={x: input_Data[minbatch_index,0:12288], y_: input_Data[minbatch_index,12288:12292], keep_prob: 1, phase: 0}))
			file3.write(str(cross_entropy.eval(feed_dict={x: input_Data[minbatch_index,0:12288], y_: input_Data[minbatch_index,12288:12292], keep_prob: 1, phase: 0})))
			file3.write("\n")
			file4.write(str(yt.eval(feed_dict={x:test_Data[:,0:12288], y_: test_Data[:,12288:12292], keep_prob: 1, phase: 0})))
			file4.write("\n")
			file5.write(str(yt_.eval(feed_dict={x:test_Data[:,0:12288], y_: test_Data[:,12288:12292], keep_prob: 1, phase: 0})))
			file5.write("\n")

file.close()
file2.close()
file3.close()
file4.close()
file5.close()
