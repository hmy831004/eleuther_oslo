CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29506 --use_env bong_data_bart_pretraining_huggingface.py
# python train.py --gradient_clip_val 1.0 \
#                 --max_epochs 50 \
# 		--output_root_dir outputs \
#                 --devices 0, \
#                 --accelerator gpu \
#                 --strategy ddp \
#                 --num_workers 0 \
#                 --replace_sampler_ddp false \
#                 --batch_size 4 \
#                 --dataset aihub \
#                 --train_file data/train_aihub_kobart_mix.tsv \
#                 --val_file data/val_kobart.tsv \
#                 --model_path ./outputs/model_bin

            
