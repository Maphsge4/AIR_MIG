# 开启mig
nvidia-smi -i 0 -mig 1

# 关闭服务
ls /proc/*/fd/* -l | grep /dev/nvidi
ls -l /proc/<proc id>/exe
sudo systemctl stop nvidia-dcgm

# 查看现在的MIG实例
nvidia-smi mig -lgip 

# 创建实例，5表示id，可替换 0 5 9 14 15 20 19
nvidia-smi mig -cgi 5 -C 

# 删除实例
sudo nvidia-smi mig -dci && sudo nvidia-smi mig -dgi

# 查看uuid
nvidia-smi -L

# 运行带宽测试程序
CUDA_VISIBLE_DEVICES=<uuid> ./bandwidthTest

