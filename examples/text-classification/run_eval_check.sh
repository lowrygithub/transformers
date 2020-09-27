#!/bin/bash
#!/bin/bash
echo "start"
while true
do
for i in /data/kerli/SpaceV/Model/Hrs/full/*/*
	do
                if [ ! -f "$i/eval_results_dssm.txt" ]
                then
			echo "$i need to run eval_results"
			if [ -e "$i-b" ]
			then
				echo "-b exists"
			fi
                fi
        done
	sleep 900
done
