#!/bin/bash
#!/bin/bash
echo "start"
while true
do
for i in /data/kerli/SpaceV/Model/Hrs/full/*/*
	do
                if [ ! -f "$i/eval_results_dssm.txt" ]
                then
			if [ -e "$i-b" ]
			then
				cp /home/kerli/data/vocab/*token*  "$i"
				cp /home/kerli/data/vocab/*token*  "$i-b"
				cp /home/kerli/data/vocab/vocab.txt  "$i"
				cp /home/kerli/data/vocab/vocab.txt  "$i-b"
				python run_dssm.py --model_name_or_path_b "$i-b"  --model_name_or_path "$i"  --task_name DSSM  --do_eval   --data_dir /multimedia-nfs/kerli/SpaceV/Data/Hrs/full/   --max_seq_length 128   --per_device_train_batch_size 64   --learning_rate 2e-5   --num_train_epochs 1  --output_dir /tmp/kerli/SpaceV/Model/Hrs/full/v0 --per_device_eval_batch_size 64
			fi
                fi
        done
	sleep 900
done
