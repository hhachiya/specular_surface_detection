# GPU setting number ex) 0
DEVICE_NUM=$1
# execute date, ex) 20230912
date=$2
write_dir=${date}
for iter in {1,2,3}; do
    # train
    CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python train.py -dataset_path data/spherical_mirror_dataset -write_dir ${write_dir}/ex_${iter} -mode rccl 
    CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python train.py -dataset_path data/spherical_mirror_dataset -write_dir ${write_dir}/ex_${iter} -mode ssf  
    CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python train.py -dataset_path data/spherical_mirror_dataset -write_dir ${write_dir}/ex_${iter} -mode sh 
    CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python train.py -dataset_path data/spherical_mirror_dataset -write_dir ${write_dir}/ex_${iter} -mode refine  
    # test
    CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python test.py -dataset_path data/spherical_mirror_dataset -write_dir ${write_dir}/ex_${iter} 
done
