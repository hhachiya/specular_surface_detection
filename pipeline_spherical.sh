# GPU setting number ex) 0
DEVICE_NUM=$1
# execute date, ex) 20230912
date=$2
write_dir=${date}
for iter in {1,2,3}; do
    CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python main.py -dataset_path data/spherical_mirror_dataset -write_dir ${write_dir}/ex_${iter} -mode rccl --train
    CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python main.py -dataset_path data/spherical_mirror_dataset -write_dir ${write_dir}/ex_${iter} -mode ssf --train 
    CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python main.py -dataset_path data/spherical_mirror_dataset -write_dir ${write_dir}/ex_${iter} -mode sh --train
    CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python main.py -dataset_path data/spherical_mirror_dataset -write_dir ${write_dir}/ex_${iter} -mode refine  --train --eval
    CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python main.py -dataset_path data/spherical_mirror_dataset -write_dir ${write_dir}/ex_${iter} -mode pmd --train --eval
done
