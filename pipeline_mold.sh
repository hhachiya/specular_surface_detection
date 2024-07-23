# GPU setting number ex) 0
DEVICE_NUM=$1
# execute date, ex) 20230912
date=$2
cd data/plastic_mold_dataset
KINDS=$(ls .); cd ../..
for kind in ${KINDS}; do
    write_dir=${date}/${kind}
    for iter in {1,2,3}; do
        # train
        CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python train.py -dataset_path data/plastic_mold_dataset -write_dir ${write_dir}/ex_${iter} -test_mold_type ${kind} -mode rccl 
        CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python train.py -dataset_path data/plastic_mold_dataset -write_dir ${write_dir}/ex_${iter} -test_mold_type ${kind} -mode ssf  
        CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python train.py -dataset_path data/plastic_mold_dataset -write_dir ${write_dir}/ex_${iter} -test_mold_type ${kind} -mode sh 
        CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python train.py -dataset_path data/plastic_mold_dataset -write_dir ${write_dir}/ex_${iter} -test_mold_type ${kind} -mode refine  
        # test
        CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python test.py -dataset_path data/plastic_mold_dataset -write_dir ${write_dir}/ex_${iter} -test_mold_type ${kind} 
        done
done