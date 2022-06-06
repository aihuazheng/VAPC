#dataset=market1501
#dataset=duke
#dataset=mars
#dataset=DukeMTMC-VideoReID

batchSize=32
size_penalty=0.003
merge_percent=0.05

logs=bottom_logs/


CUDA_VISIBLE_DEVICES=3,4 python run.py  --logs_dir $logs \
              -b $batchSize --size_penalty $size_penalty -mp $merge_percent 
