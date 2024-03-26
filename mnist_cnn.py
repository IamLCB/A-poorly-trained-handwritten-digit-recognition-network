import keras
import tensorflow as tf
import os
from keras.datasets import mnist
from keras import layers, models
from tensorflow.keras.models import load_model

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 设置训练轮数
train_round = 100

# 设置采用GPU训练程序
gpus = tf.config.list_physical_devices("GPU")  # 获取电脑GPU列表
if gpus:  # gpus不为空
    gpu0 = gpus[0]  # 选取GPU列表中的第一个
    tf.config.experimental.set_memory_growth(gpu0, True)  # 设置GPU显卡按需使用
    tf.config.set_visible_devices([gpu0], "GPU")  # 设置GPU可见的设备清单，默认是都可见，这里只设置了gpu0可见

# 导入数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()  # datasets内部集成了MNIST数据集，
print(train_images.shape, train_labels.shape)

# 归一化
# 将像素的值标准化至0到1的区间内,rgb像素值 0~255 0为黑 1为白
train_images, test_images = train_images / 255.0, test_images / 255.0

# 根据数据集大小调整数据到我们需要的格式
print(train_images.size)  # 47040000
print(test_images.size)  # 7840000
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# 检查模型文件是否存在，如果存在，则加载模型继续训练；否则，创建新模型

model_file_path = 'models/mnist_model.h5'
if os.path.exists(model_file_path):
    model = load_model(model_file_path)
    print("Loaded model from disk.")
else:
    # 构建CNN网络模型
    model = models.Sequential([  # 采用Sequential 顺序模型
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        # 卷积层1，卷积核个数32，卷积核3*3*1 relu激活去除负值保留正值，输入是28*28*1
        layers.MaxPooling2D((2, 2)),  # 池化层1，2*2采样
        layers.Conv2D(64, (3, 3), activation='relu'),
        # 卷积层2，卷积核64个，卷积核3*3，relu激活去除负值保留正值
        layers.MaxPooling2D((2, 2)),  # 池化层2，2*2采样
        layers.Dropout(0.25),  # Dropout层，防止过拟合
        layers.Flatten(),  # Flatten层，连接卷积层与全连接层
        layers.Dense(256, activation='relu'),  # 全连接层，256张特征图，特征进一步提取
        layers.Dropout(0.5),  # Dropout层，防止过拟合
        layers.Dense(10, activation='softmax')  # 全连接层，输出层，10个神经元，softmax激活函数
    ])
    print("Created new model.")
# 打印网络结构
model.summary()

# 编译
model.compile(optimizer='adam',  # 优化器
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              # 设置损失函数from_logits: 为True时，会将y_pred转化为概率
              metrics=['accuracy'])

# 训练
# epochs为训练轮数
# batch_size为每次训练的数据批次

history = model.fit(train_images, train_labels, batch_size=128, epochs=train_round,
                    validation_data=(test_images, test_labels))
print(history)

model.save(model_file_path)

score = model.evaluate(test_images, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


def load_trained_model(model_path):
    return load_model(model_path)
