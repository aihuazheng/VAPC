参考文献https://github.com/vana77/Bottom-up-Clustering-Person-Re-identification
主要改动原始bottom_up.py，其他部分基本一样，改动部分另命名为veri.py 550行左右开始
训练和测试均直接执行sh run.sh
models/ 网络模型
logs/ 一些预训练模型
evaluaters.py 测试代码
exclusive_loss.py 损失函数
veri.py 渐进式聚类代码
viewtrain.py 视角预测和训练代码
env36.yml 训练环境