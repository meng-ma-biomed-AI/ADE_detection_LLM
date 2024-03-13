module load Sali
module load gcc
export PATH=$PATH:~/anaconda3/bin
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ucsf-ade
module load cuda/11.5.0
export PYTHONPATH=~/ade:$PYTHONPATH

TASK="ucsf_hosp_for_ae"
DATADIR="/wynton/protected/project/outcome_pred/ade/data/"
MODEL_TYPE="bert"
CHECKPOINT_DIR="/wynton/protected/project/outcome_pred/ucsf_bert_pytorch/512/500k-275k/"
MAX_SEQ_LEN=2560
DECODER="fcn"
run=0
for seed in {10,20,30,40,50}; do
  (( run++ ))
  OUTPUT_DIR="/wynton/protected/project/outcome_pred/results/${TASK}/hier/us_0.8/bert/ucsf_bert-512-500k+275k/run${run}/"
  CUDA_VISIBLE_DEVICES=$SGE_GPU python -m run_transformers_classification \
    --task_name ${TASK}\
    --data_dir ${DATADIR}\
    --model_type ${MODEL_TYPE}\
    --model_name_or_path ${CHECKPOINT_DIR}\
    --tokenizer_name ${CHECKPOINT_DIR}\
    --output_dir ${OUTPUT_DIR} \
    --max_seq_length ${MAX_SEQ_LEN}\
    --num_train_epochs 10\
    --per_gpu_train_batch_size 8\
    --per_gpu_eval_batch_size 8\
    --save_steps 100\
    --seed ${seed}\
    --gradient_accumulation_steps 4\
    --learning_rate 2e-5\
    --do_train\
    --do_eval\
    --hierarchical\
    --decoder ${DECODER}\
    --warmup_steps 0\
    --overwrite_output_dir \
    --overwrite_cache

  CUDA_VISIBLE_DEVICES=$SGE_GPU python -m ade.run_transformers_classification \
    --task_name ${TASK}\
    --data_dir ${DATADIR}\
    --model_type ${MODEL_TYPE}\
    --model_name_or_path ${CHECKPOINT_DIR}\
    --tokenizer_name ${CHECKPOINT_DIR}\
    --output_dir ${OUTPUT_DIR} \
    --max_seq_length ${MAX_SEQ_LEN}\
    --per_gpu_eval_batch_size 8\
    --do_test\
    --hierarchical\
    --overwrite_output_dir \
    --overwrite_cache
done